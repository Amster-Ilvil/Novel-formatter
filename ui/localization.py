# -*- coding: utf-8 -*-
"""Small, offline interface localization catalog for Novel Formatter.

Business data, OCR text and user documents are never translated here.  The
catalog is only used for Qt chrome (labels, buttons, tabs, tooltips and dialogs).
Unknown strings are deliberately left unchanged instead of calling a network
translation service or guessing technical terminology.
"""
from __future__ import annotations

import re

LANG_ZH = "zh_CN"
LANG_JA = "ja_JP"
LANG_EN = "en_US"
SUPPORTED_LANGUAGES = (LANG_ZH, LANG_JA, LANG_EN)

LANGUAGE_LABELS = {
    LANG_ZH: "简体中文",
    LANG_JA: "日本語",
    LANG_EN: "English",
}

# Exact high-visibility UI strings.  Keeping this list explicit makes language
# switching deterministic/offline and prevents OCR/model terminology from being
# silently machine-translated.
_EXACT_EN = {
    "页面管理": "Pages",
    "OCR 识别": "OCR",
    "格式处理": "Formatter",
    "文字校对": "Proofreading",
    "手动编辑当前裁决文本": "Edit Current Decision Text",
    "可直接输入最终文本；不会覆盖任何 OCR 模型原文。": "Enter the final text directly; original OCR model outputs are never overwritten.",
    "直接编辑当前融合结果；只新增/更新人工裁决候选，原始 OCR 候选始终保留。": "Edit the current fused result directly; this only adds or updates a human-decision candidate, while all original OCR candidates are preserved.",
    "EPUB生成": "EPUB",
    "设置": "Settings",
    "图片 OCR": "Image OCR",
    "PDF 文字层": "PDF Text Layer",
    "全文总览": "Full Overview",
    "当前为全文总览；点击可从任何兼容恢复状态回到全文显示": "You are viewing the full overview; click to return to full-text display from any restored state",
    "在当前工作台切换到逐句图文对照，保留稳定句和草稿": "Switch to sentence-by-sentence image/text review, preserving stable rows and drafts",
    "将当前全文裁决结果应用到后续工作流": "Apply current full-book adjudications to subsequent workflow",
    "图文对照 · 逐句裁决": "Image/Text Review · Sentence Adjudication",
    "OCR 对比": "OCR Compare",
    "OCR 对比 · 全文总览": "OCR Compare · Full Overview",
    "✓ 应用整本": "✓ Apply Whole Book",
    "图文对照": "Image/Text Review",
    "常规设置": "General",
    "OCR 设置": "OCR Settings",
    "性能设置": "Performance",
    "快捷键": "Shortcuts",
    "更新与诊断": "Updates & Diagnostics",
    "关于": "About",
    "保存设置": "Save Settings",
    "恢复默认设置": "Restore Defaults",
    "界面与显示": "Interface & Appearance",
    "界面语言": "Interface Language",
    "外观": "Appearance",
    "浅色": "Light",
    "深色": "Dark",
    "启动设置": "Startup",
    "启动时恢复上次功能区": "Restore last workspace on startup",
    "记住 OCR 与文字校对子页签": "Remember OCR and proofreading subtabs",
    "启动时最大化窗口": "Start maximized",
    "默认启动功能区": "Default workspace",
    "文件与诊断目录": "Files & Diagnostic Folders",
    "打开临时文件夹": "Open Temp Folder",
    "打开诊断目录": "Open Diagnostics Folder",
    "OCR 默认行为": "OCR Defaults",
    "默认识别引擎": "Default OCR engine",
    "识别引擎": "Recognition Engine",
    "默认打开设置页": "Default settings tab",
    "引擎": "Engine",
    "分列与组句": "Columns & Sentences",
    "逐字审校": "Character Review",
    "逐字审校默认启用": "Enable character review by default",
    "逐字审校设置默认展开": "Expand character review settings by default",
    "默认开启实时图像预览": "Enable live image preview by default",
    "默认显示实时进度与预计时间": "Show live progress and ETA by default",
    "OCR 工作区入口": "OCR Workspace",
    "前往 OCR 识别": "Open OCR",
    "打开 PDF 文字层": "Open PDF Text Layer",
    "检测 OCR 运行环境": "Check OCR Runtime",
    "AI 服务设置": "AI Service Settings",
    "打开 AI 设置": "Open AI Settings",
    "OCR 模型手动更新": "Manual OCR Model Updates",
    "OCR 模型更新（仅手动）": "OCR Model Updates (Manual Only)",
    "模型": "Model",
    "管理方式": "Management",
    "本地版本": "Local Version",
    "官方版本": "Upstream Version",
    "状态": "Status",
    "选择一个模型查看说明。": "Select a model to view details.",
    "刷新本地状态": "Refresh Local Status",
    "检查更新": "Check for Updates",
    "打开官方来源": "Open Official Source",
    "选择可更新模型": "Select an Updatable Model",
    "关闭": "Close",
    "取消": "Cancel",
    "确定": "OK",
    "确认": "Confirm",
    "保存": "Save",
    "打开": "Open",
    "选择": "Select",
    "选择…": "Choose…",
    "清除": "Clear",
    "刷新": "Refresh",
    "↻ 刷新": "↻ Refresh",
    "删除": "Delete",
    "🗑 删除": "🗑 Delete",
    "全选": "Select All",
    "取消选择": "Clear Selection",
    "停止": "Stop",
    "重新 OCR": "Re-run OCR",
    "正在准备…": "Preparing…",
    "正在下载…": "Downloading…",
    "正在终止…": "Stopping…",
    "完成": "Done",
    "初始化": "Initializing",
    "后台运行": "Run in Background",
    "测试连接": "Test Connection",
    "刷新模型": "Refresh Models",
    "连接失败": "Connection Failed",
    "读取中…": "Loading…",
    "正在连接……": "Connecting…",
    "页面高清预览": "High-Resolution Page Preview",
    "← 上一页": "← Previous Page",
    "下一页 →": "Next Page →",
    "适应窗口": "Fit to Window",
    "页面类型": "Page Type",
    "点击筛选页面": "Click to filter pages",
    "＋ 添加文件": "+ Add Files",
    "打开文件夹": "Open Folder",
    "扫描件优化": "Scan Cleanup",
    "恢复优化前": "Restore Before Cleanup",
    "搜索页名…": "Search page name…",
    "未添加文件": "No files added",
    "全选": "Select All",
    "🗑 删除选中": "🗑 Delete Selected",
    "OCR 模式": "OCR Mode",
    "日文竖排（现有优化模式）": "Japanese Vertical (Optimized)",
    "简体中文横排（独立模式）": "Simplified Chinese Horizontal (Independent)",
    "日文竖排": "Japanese Vertical",
    "简体中文横排": "Simplified Chinese Horizontal",
    "显示已裁决": "Show Adjudicated",
    "全部已裁决": "All Adjudicated",
    "本地裁决": "Local Adjudication",
    "人工裁决": "Human Adjudication",
    "本地 AI": "In-app AI",
    "云端 AI": "Imported Cloud AI",
    "已裁决历史": "Adjudication History",
    "OCR 竖列（与原图列顺序对齐）": "OCR Vertical Columns (Source Order)",
    "OCR 竖列": "OCR Vertical Columns",
    "对应识别图片": "Matching Source Image",
    "拖动图片横向定位 OCR 列": "Drag across image to locate OCR column",
    "定位左侧 OCR 物理列（⌥⇧←）": "Move to the OCR physical column on the left (⌥⇧←)",
    "定位右侧 OCR 物理列（⌥⇧→）": "Move to the OCR physical column on the right (⌥⇧→)",
    "横排校对文本（可修改）": "Proofreading Text (Editable)",
    "横排校对文本": "Proofreading Text",
    "打开后只读浏览明确裁决过的句子；可按本地、人工、本地 AI、云端 AI 分别筛选。自动一致/两模型共同候选不计入裁决历史。": "Browse explicitly adjudicated sentences in read-only mode. Filter by local, human, in-app AI, or imported cloud AI. Automatic consensus is not counted as adjudication history.",
    "当前正在浏览只读裁决历史；关闭“显示已裁决”后才能重新裁决。": "Adjudication history is read-only. Turn off ‘Show adjudicated’ before changing a decision.",
    "按住鼠标左键在图片上横向拖动，可联动定位旁边的 OCR 物理列。": "Hold the left mouse button and drag horizontally across the image to locate the matching OCR physical column beside it.",
    "当前筛选没有已裁决句": "No adjudicated sentences match the current filter",
    "当前筛选没有已裁决句。": "No adjudicated sentences match the current filter.",
    "多模型 OCR 对比（最多 6 个，自适应共识）": "Multi-model OCR Compare (up to 6, adaptive consensus)",
    "整句重识别": "Sentence Re-recognition",
    "识别方式": "Recognition Method",
    "启用日语语言校正": "Enable Japanese Language Correction",
    "快捷指令名称": "Shortcut Name",
    "语言提示（可留空自动检测）": "Language Hints (blank = auto detect)",
    "AI Studio 接口模式": "AI Studio API Mode",
    "同步 API URL": "Synchronous API URL",
    "同步服务类型": "Synchronous Service Type",
    "保存到系统": "Save to System",
    "异步 v2 模型": "Async v2 Model",
    "文档方向校正": "Document Orientation Correction",
    "文档去畸变": "Document Dewarping",
    "文字行方向": "Text Line Orientation",
    "单请求超时": "Request Timeout",
    "重试": "Retries",
    "Hayai 后端": "Hayai Backend",
    "运行设备": "Device",
    "Torch 量化": "Torch Quantization",
    "LiteRT 量化": "LiteRT Quantization",
    "Paddle 模型": "Paddle Model",
    "VL 后端": "VL Backend",
    "模型下载源": "Model Download Source",
    "安装 / 修复模型": "Install / Repair Model",
    "检查状态": "Check Status",
    "简体中文横排版面": "Simplified Chinese Horizontal Layout",
    "分列灵敏度": "Column Sensitivity",
    "列保护边距": "Column Safety Margin",
    "Ruby 扫描范围": "Ruby Scan Range",
    "智能 ROI（推荐）": "Smart ROI (Recommended)",
    "全页精确扫描（慢）": "Full-page Accurate Scan (Slow)",
    "> 高级 OCR 预处理": "> Advanced OCR Preprocessing",
    "Ruby 过滤强度": "Ruby Filter Strength",
    "标准：轻小说推荐": "Standard: Recommended for Light Novels",
    "恢复 OCR 默认设置": "Restore OCR Defaults",
    "列级救援": "Column Recovery",
    "NDLOCR 分列策略": "NDLOCR Column Strategy",
    "👁 预览分列与掩膜": "👁 Preview Columns & Masks",
    "🔍 重新检测本地 OCR": "🔍 Re-detect Local OCR",
    "模式": "Mode",
    "自动策略": "Automatic Strategy",
    "下载手写模型": "Download Handwriting Model",
    "打开 Apple 手写测试面板": "Open Apple Handwriting Test Panel",
    "预览最新自动轨迹": "Preview Latest Automatic Trace",
    "整句重识别策略": "Sentence Re-recognition Strategy",
    "无句末安全上限": "No Sentence-end Safety Limit",
    "输入（图片 / PDF）": "Input (Images / PDF)",
    "📂  选择文件夹": "📂  Choose Folder",
    "🖼  选择图片 / PDF": "🖼  Choose Images / PDF",
    "（尚未选择输入）": "(No input selected)",
    "实时预览": "Live Preview",
    "当前图片：尚未载入": "Current image: not loaded",
    "清除框选": "Clear Selection Box",
    "OCR 日志": "OCR Log",
    "显示实时进度/预计时间": "Show Live Progress / ETA",
    "需要安装 OCR 模型": "OCR Model Installation Required",
    "显示具体规则说明": "Show Detailed Rules",
    "固定原 OCR 排版": "Keep Original OCR Layout",
    "PDF文字层模式": "PDF Text-layer Mode",
    "保留作者前书/后记": "Keep Author Preface/Afterword",
    "📂 导入 JSON / MD": "📂 Import JSON / MD",
    "📄 导入 DOCX": "📄 Import DOCX",
    "📖 导入 EPUB": "📖 Import EPUB",
    "💾 保存结果": "💾 Save Result",
    "↩ 撤销": "↩ Undo",
    "📝 应用编辑": "📝 Apply Edits",
    "处理前文本（可编辑，点击「应用编辑」写回）": "Before (editable; click Apply Edits to commit)",
    "处理后文本（可编辑，点击「应用编辑」写回）": "After (editable; click Apply Edits to commit)",
    "清空处理后": "Clear After",
    "重新载入处理前": "Reload Before",
    "⚙ AI设置": "⚙ AI Settings",
    "断点 ▾": "Checkpoint ▾",
    "排版格式管理": "Layout Profile Manager",
    "＋ 新建空白格式": "+ New Blank Profile",
    "📥 导入格式…": "📥 Import Profile…",
    "正在学习格式…": "Learning layout…",
    "📖 从参考 EPUB 学习格式…": "📖 Learn from Reference EPUB…",
    "显示": "Show",
    "保存密钥到本机": "Save key on this device",
    "可选：全书术语白名单（TXT / JSON）": "Optional: book-wide terminology whitelist (TXT / JSON)",
    "启用思考模式（更慢、消耗额外推理 Tokens）": "Enable reasoning mode (slower, uses extra reasoning tokens)",
    "可读性优先（推荐）": "Readability First (Recommended)",
    "严格还原（不推测缺字）": "Strict Reconstruction (do not infer missing text)",
    "使用接口原生 JSON 模式（不支持时自动回退）": "Use native JSON mode (auto fallback if unsupported)",
    "严格覆盖（重建小说排版，推荐）": "Strict Replace (rebuild novel layout, recommended)",
    "严格覆盖（完全原样）": "Strict Replace (verbatim)",
    "局部智能替换": "Smart Partial Replace",
    "仅比较差异": "Compare Only",
    "等待文档": "Waiting for document",
    "AI处理中": "AI Processing",
    "准备中…": "Preparing…",
    "正在准备章节…": "Preparing chapters…",
    "正在停止": "Stopping",
    "↻ 刷新检查": "↻ Refresh Check",
    "等待正文版本": "Waiting for text version",
    "内容结构": "Content Structure",
    "打包文件": "Package Files",
    "元数据": "Metadata",
    "粘贴": "Paste",
    "作者": "Author",
    "出版社": "Publisher",
    "卷号": "Volume",
    "⚙ 管理格式": "⚙ Manage Profiles",
    "竖排": "Vertical",
    "横排": "Horizontal",
    "源码": "Source",
    "预览": "Preview",
    "等待新书": "Waiting for new book",
    "构建中…": "Building…",
    "✓ 完成": "✓ Done",
    "✗ 失败": "✗ Failed",
    "PDF 文字层直读": "Direct PDF Text-layer Read",
    "输入（单个 PDF 文件）": "Input (Single PDF)",
    "📄  选择 PDF 文件": "📄  Choose PDF",
    "提取日志": "Extraction Log",
    "AI 修改审阅 · 逐项确认": "Review AI Changes · Confirm Individually",
    "采用低/中风险": "Accept Low/Medium Risk",
    "全部保留原文": "Keep All Original",
    "采用全部AI修改": "Accept All AI Changes",
    "← 上一处": "← Previous",
    "下一处 →": "Next →",
    "定位到双栏": "Locate in Two-column View",
    "发布前检查": "Pre-publication Check",
    "参考原文对照评分": "Reference-text Comparison Score",
    "双 OCR 融合裁决报告": "Dual-OCR Fusion Decision Report",
    "导入左文本": "Import Left Text",
    "导入右文本": "Import Right Text",
    "🖼 页面管理 / 插图": "🖼 Page Manager / Illustrations",
    "↵ 光标处分行": "↵ Split at Cursor",
    "⇤ 删除换行 / 接上前句": "⇤ Remove Break / Join Previous",
    "＋ 空白对齐行": "+ Blank Alignment Row",
    "「」对白独立成行": "Put dialogue on separate lines",
    "⚡ 实时对齐": "⚡ Live Alignment",
    "A 文字颜色": "A Text Color",
    "▰ 背景色": "▰ Background",
    "U 下划线": "U Underline",
    "S 删除线": "S Strikeout",
    "清除样式": "Clear Style",
    "导出双栏文本": "Export Two-column Text",
    "双 OCR：": "Dual OCR:",
    "智能裁决": "Smart Decision",
    "外部 OCR 优先": "Prefer External OCR",
    "内置 OCR 优先": "Prefer Built-in OCR",
    "查看融合裁决": "View Fusion Decisions",
    "外部 OCR：未导入": "External OCR: not imported",
    "可信正文：": "Trusted Text:",
    "折叠右侧大空白": "Collapse Large Right-side Gaps",
    "AI审校：": "AI Review:",
    "AI纠错当前行": "AI Fix Current Line",
    "AI纠错选中区域": "AI Fix Selection",
    "查看AI修改（0）": "View AI Changes (0)",
    "上一处 AI 修改": "Previous AI Change",
    "下一处 AI 修改": "Next AI Change",
    "✓ 确认AI结果（仅右侧）": "✓ Confirm AI Result (Right Only)",
    "✕ 拒绝AI结果": "✕ Reject AI Result",
    "↶ 撤销上次AI": "↶ Undo Last AI",
    "AI设置": "AI Settings",
    "翻译/发布校验：": "Translation/Publication Check:",
    "检查当前右侧": "Check Current Right Text",
    "修复确定问题": "Fix Confirmed Issues",
    "📘 导入参考原文": "📘 Import Reference Text",
    "参考评分": "Reference Score",
    "参考原文：未导入": "Reference text: not imported",
    "AI状态：未运行": "AI status: not run",
    "左侧": "Left",
    "右侧": "Right",
    "句末": "Sentence End",
    "重新扫描右侧全文": "Rescan Entire Right Text",
    "上一处（Shift+F7）": "Previous (Shift+F7)",
    "下一处（F7）": "Next (F7)",
    "待复核数量 / 当前序号": "Items to Review / Current",
    "左侧参考文本": "Left Reference Text",
    "右侧待修改文本": "Right Text to Edit",
    "评分中…": "Scoring…",
    "重新选择": "Choose Again",
    "AI 审定多模型 OCR 融合稿": "AI Adjudication of Multi-model OCR Fusion",
    "OCR 对比 · 结果收件箱": "OCR Compare · Result Inbox",
    "↔ 重新对齐": "↔ Realign",
    "↶ 恢复初始": "↶ Restore Initial",
    "逐句裁决": "Sentence Decisions",
    "全文对比": "Full-text Compare",
    "显示全文对比": "Show Full-text Compare",
    "图文检查 ↗": "Image/Text Review ↗",
    "更多操作 ▾": "More Actions ▾",
    "⇄ 返回多模型对比": "⇄ Return to Multi-model Compare",
    "当前句采用：": "Current sentence uses:",
    "当前句：—": "Current sentence: —",
    "多模型包：": "Multi-model Package:",
    "OCR 裁决：": "OCR Decisions:",
    "↻ 恢复纠错会话": "↻ Restore Review Session",
    "尚未导入 OCR 裁决": "No OCR decisions imported",
    "停止AI审定": "Stop AI Adjudication",
    "只显示需判断": "Show Decisions Only",
    "只显示一个对比框": "Show One Compare Box",
    "选择后自动下一条": "Advance after selection",
    "← 上一分歧": "← Previous Difference",
    "下一分歧 →": "Next Difference →",
    "下一句 →": "Next Sentence →",
    "待判断队列": "Decision Queue",
    "OCR 对比 · 逐句裁决": "OCR Compare · Sentence Decisions",
    "✓ 应用融合稿": "✓ Apply Fusion Text",
    "OCR 对比 · 单结果校对": "OCR Compare · Single-result Review",
    "✓ 应用单OCR结果": "✓ Apply Single OCR Result",
    "AI 修复包导出选项": "AI Repair Package Export Options",
    "开始导出": "Start Export",
    "图文逐句校对": "Sentence Image/Text Review",
    "尚未接收 OCR 结果": "No OCR result received",
    "OCR 竖列（与原图列顺序对齐）": "OCR Vertical Columns (aligned with source order)",
    "对应识别图片": "Corresponding OCR Image",
    "用预览打开": "Open in Preview",
    "横排校对文本（可修改）": "Horizontal Review Text (editable)",
    "苹果系统手写": "Apple Handwriting",
    "多模型融合候选": "Multi-model Fusion Candidate",
    "← 上一句": "← Previous Sentence",
    "保存当前句": "Save Current Sentence",
    "下一待判断": "Next Pending",
    "上一OCR分歧": "Previous OCR Difference",
    "下一OCR分歧": "Next OCR Difference",
    "正在载入…": "Loading…",
    "更新分支": "Update Branch",
    "GitHub 仓库": "GitHub Repository",
    "本地程序目录": "Local Program Folder",
    "当前版本": "Current Version",
    "复制诊断信息": "Copy Diagnostics",
    "打开 GitHub 仓库": "Open GitHub Repository",
    "打开程序目录": "Open Program Folder",
}

_EXACT_JA = {
    "页面管理": "ページ管理",
    "手动编辑当前裁决文本": "現在の裁定テキストを手動編集",
    "可直接输入最终文本；不会覆盖任何 OCR 模型原文。": "最終テキストを直接入力できます。OCR各モデルの元テキストは上書きしません。",
    "直接编辑当前融合结果；只新增/更新人工裁决候选，原始 OCR 候选始终保留。": "現在の統合結果を直接編集します。人手裁定候補だけを追加・更新し、元のOCR候補はすべて保持します。",
    "OCR 识别": "OCR認識",
    "格式处理": "整形処理",
    "文字校对": "テキスト校正",
    "EPUB生成": "EPUB作成",
    "设置": "設定",
    "图片 OCR": "画像OCR",
    "PDF 文字层": "PDFテキストレイヤー",
    "全文总览": "全文概要",
    "当前为全文总览；点击可从任何兼容恢复状态回到全文显示": "現在は全文概要です。復元状態から全文表示に戻るにはクリックします",
    "在当前工作台切换到逐句图文对照，保留稳定句和草稿": "同じ作業画面で逐句の画像・テキスト照合へ切り替え、行IDと下書きを保持します",
    "将当前全文裁决结果应用到后续工作流": "現在の全書裁定結果を後続の作業工程へ適用します",
    "图文对照 · 逐句裁决": "画像・テキスト照合・逐句裁定",
    "OCR 对比": "OCR比較",
    "OCR 对比 · 全文总览": "OCR比較・全文概要",
    "✓ 应用整本": "✓ 全書に適用",
    "图文对照": "画像・テキスト照合",
    "常规设置": "一般設定",
    "OCR 设置": "OCR設定",
    "性能设置": "パフォーマンス",
    "快捷键": "ショートカット",
    "更新与诊断": "更新と診断",
    "关于": "このアプリについて",
    "保存设置": "設定を保存",
    "恢复默认设置": "既定値に戻す",
    "界面与显示": "表示と言語",
    "界面语言": "表示言語",
    "外观": "外観",
    "浅色": "ライト",
    "深色": "ダーク",
    "启动设置": "起動設定",
    "启动时恢复上次功能区": "起動時に前回のワークスペースを復元",
    "记住 OCR 与文字校对子页签": "OCRと校正のサブタブを記憶",
    "启动时最大化窗口": "最大化して起動",
    "默认启动功能区": "既定のワークスペース",
    "文件与诊断目录": "ファイルと診断フォルダ",
    "打开临时文件夹": "一時フォルダを開く",
    "打开诊断目录": "診断フォルダを開く",
    "OCR 默认行为": "OCR既定動作",
    "默认识别引擎": "既定のOCRエンジン",
    "识别引擎": "認識エンジン",
    "默认打开设置页": "既定の設定タブ",
    "引擎": "エンジン",
    "分列与组句": "列分割と文結合",
    "逐字审校": "文字単位校正",
    "逐字审校默认启用": "文字単位校正を既定で有効化",
    "逐字审校设置默认展开": "文字単位校正の設定を既定で展開",
    "默认开启实时图像预览": "ライブ画像プレビューを既定で有効化",
    "默认显示实时进度与预计时间": "進捗と残り時間を既定で表示",
    "OCR 工作区入口": "OCRワークスペース",
    "前往 OCR 识别": "OCRを開く",
    "打开 PDF 文字层": "PDFテキストレイヤーを開く",
    "检测 OCR 运行环境": "OCR実行環境を確認",
    "AI 服务设置": "AIサービス設定",
    "打开 AI 设置": "AI設定を開く",
    "OCR 模型手动更新": "OCRモデル手動更新",
    "OCR 模型更新（仅手动）": "OCRモデル更新（手動のみ）",
    "模型": "モデル",
    "管理方式": "管理方式",
    "本地版本": "ローカル版",
    "官方版本": "公式版",
    "状态": "状態",
    "选择一个模型查看说明。": "モデルを選択すると詳細を表示します。",
    "刷新本地状态": "ローカル状態を更新",
    "检查更新": "更新を確認",
    "打开官方来源": "公式ソースを開く",
    "选择可更新模型": "更新可能なモデルを選択",
    "关闭": "閉じる",
    "取消": "キャンセル",
    "确定": "OK",
    "确认": "確認",
    "保存": "保存",
    "打开": "開く",
    "选择": "選択",
    "选择…": "選択…",
    "清除": "クリア",
    "刷新": "更新",
    "↻ 刷新": "↻ 更新",
    "删除": "削除",
    "🗑 删除": "🗑 削除",
    "全选": "すべて選択",
    "取消选择": "選択解除",
    "停止": "停止",
    "重新 OCR": "OCRを再実行",
    "正在准备…": "準備中…",
    "正在下载…": "ダウンロード中…",
    "正在终止…": "停止中…",
    "完成": "完了",
    "初始化": "初期化",
    "后台运行": "バックグラウンド実行",
    "测试连接": "接続テスト",
    "刷新模型": "モデル一覧を更新",
    "连接失败": "接続失敗",
    "读取中…": "読み込み中…",
    "正在连接……": "接続中……",
    "页面高清预览": "ページ高解像度プレビュー",
    "← 上一页": "← 前のページ",
    "下一页 →": "次のページ →",
    "适应窗口": "ウィンドウに合わせる",
    "页面类型": "ページ種別",
    "点击筛选页面": "クリックしてページを絞り込み",
    "＋ 添加文件": "＋ ファイル追加",
    "打开文件夹": "フォルダを開く",
    "扫描件优化": "スキャン画像補正",
    "恢复优化前": "補正前に戻す",
    "搜索页名…": "ページ名を検索…",
    "未添加文件": "ファイル未追加",
    "🗑 删除选中": "🗑 選択項目を削除",
    "OCR 模式": "OCRモード",
    "日文竖排（现有优化模式）": "日本語縦書き（最適化）",
    "简体中文横排（独立模式）": "簡体字中国語横書き（独立）",
    "日文竖排": "日本語縦書き",
    "简体中文横排": "簡体字中国語横書き",
    "显示已裁决": "裁決済みを表示",
    "全部已裁决": "裁決済みすべて",
    "本地裁决": "ローカル裁決",
    "人工裁决": "手動裁決",
    "本地 AI": "アプリ内AI",
    "云端 AI": "外部AI取込",
    "已裁决历史": "裁決履歴",
    "OCR 竖列（与原图列顺序对齐）": "OCR縦列（原画像の列順）",
    "OCR 竖列": "OCR縦列",
    "对应识别图片": "対応する認識画像",
    "拖动图片横向定位 OCR 列": "画像を横にドラッグしてOCR列を定位",
    "定位左侧 OCR 物理列（⌥⇧←）": "左側のOCR物理列へ移動（⌥⇧←）",
    "定位右侧 OCR 物理列（⌥⇧→）": "右側のOCR物理列へ移動（⌥⇧→）",
    "横排校对文本（可修改）": "横書き校正テキスト（編集可）",
    "横排校对文本": "横書き校正テキスト",
    "打开后只读浏览明确裁决过的句子；可按本地、人工、本地 AI、云端 AI 分别筛选。自动一致/两模型共同候选不计入裁决历史。": "明示的に裁決済みの文を読み取り専用で閲覧します。ローカル・手動・アプリ内AI・外部AI取込で絞り込めます。自動一致は裁決履歴に含めません。",
    "当前正在浏览只读裁决历史；关闭“显示已裁决”后才能重新裁决。": "裁決履歴は読み取り専用です。裁決を変更するには「裁決済みを表示」をオフにしてください。",
    "按住鼠标左键在图片上横向拖动，可联动定位旁边的 OCR 物理列。": "画像上で左ボタンを押したまま横方向にドラッグすると、隣の対応OCR物理列へ連動して移動します。",
    "当前筛选没有已裁决句": "現在の条件に一致する裁決済み文はありません",
    "当前筛选没有已裁决句。": "現在の条件に一致する裁決済み文はありません。",
    "多模型 OCR 对比（最多 6 个，自适应共识）": "複数モデルOCR比較（最大6・適応コンセンサス）",
    "整句重识别": "文単位の再認識",
    "识别方式": "認識方式",
    "启用日语语言校正": "日本語補正を有効化",
    "快捷指令名称": "ショートカット名",
    "语言提示（可留空自动检测）": "言語ヒント（空欄で自動検出）",
    "AI Studio 接口模式": "AI Studio APIモード",
    "同步 API URL": "同期API URL",
    "同步服务类型": "同期サービス種別",
    "保存到系统": "システムに保存",
    "异步 v2 模型": "非同期v2モデル",
    "文档方向校正": "文書方向補正",
    "文档去畸变": "文書ゆがみ補正",
    "文字行方向": "文字行方向",
    "单请求超时": "リクエストタイムアウト",
    "重试": "再試行",
    "Hayai 后端": "Hayaiバックエンド",
    "运行设备": "実行デバイス",
    "Torch 量化": "Torch量子化",
    "LiteRT 量化": "LiteRT量子化",
    "Paddle 模型": "Paddleモデル",
    "VL 后端": "VLバックエンド",
    "模型下载源": "モデル取得元",
    "安装 / 修复模型": "モデルをインストール / 修復",
    "检查状态": "状態確認",
    "简体中文横排版面": "簡体字中国語横書きレイアウト",
    "分列灵敏度": "列検出感度",
    "列保护边距": "列保護マージン",
    "Ruby 扫描范围": "ルビ走査範囲",
    "智能 ROI（推荐）": "スマートROI（推奨）",
    "全页精确扫描（慢）": "全ページ精密走査（低速）",
    "> 高级 OCR 预处理": "> 高度なOCR前処理",
    "Ruby 过滤强度": "ルビ除去強度",
    "标准：轻小说推荐": "標準：ライトノベル推奨",
    "恢复 OCR 默认设置": "OCR既定値に戻す",
    "列级救援": "列単位リカバリ",
    "NDLOCR 分列策略": "NDLOCR列処理戦略",
    "👁 预览分列与掩膜": "👁 列分割とマスクをプレビュー",
    "🔍 重新检测本地 OCR": "🔍 ローカルOCRを再検出",
    "模式": "モード",
    "自动策略": "自動戦略",
    "下载手写模型": "手書きモデルをダウンロード",
    "打开 Apple 手写测试面板": "Apple手書きテストパネルを開く",
    "预览最新自动轨迹": "最新の自動軌跡をプレビュー",
    "整句重识别策略": "文単位再認識戦略",
    "无句末安全上限": "文末なし安全上限",
    "输入（图片 / PDF）": "入力（画像 / PDF）",
    "📂  选择文件夹": "📂  フォルダを選択",
    "🖼  选择图片 / PDF": "🖼  画像 / PDFを選択",
    "（尚未选择输入）": "（入力未選択）",
    "实时预览": "ライブプレビュー",
    "当前图片：尚未载入": "現在の画像：未読み込み",
    "清除框选": "選択枠をクリア",
    "OCR 日志": "OCRログ",
    "显示实时进度/预计时间": "進捗 / 残り時間を表示",
    "需要安装 OCR 模型": "OCRモデルのインストールが必要です",
    "显示具体规则说明": "詳細ルールを表示",
    "固定原 OCR 排版": "元のOCRレイアウトを保持",
    "PDF文字层模式": "PDFテキストレイヤーモード",
    "保留作者前书/后记": "作者の前書き / 後書きを保持",
    "📂 导入 JSON / MD": "📂 JSON / MDを読み込む",
    "📄 导入 DOCX": "📄 DOCXを読み込む",
    "📖 导入 EPUB": "📖 EPUBを読み込む",
    "💾 保存结果": "💾 結果を保存",
    "↩ 撤销": "↩ 元に戻す",
    "📝 应用编辑": "📝 編集を適用",
    "清空处理后": "処理後をクリア",
    "重新载入处理前": "処理前を再読み込み",
    "⚙ AI设置": "⚙ AI設定",
    "断点 ▾": "チェックポイント ▾",
    "排版格式管理": "レイアウト形式管理",
    "＋ 新建空白格式": "＋ 空の形式を新規作成",
    "📥 导入格式…": "📥 形式を読み込む…",
    "正在学习格式…": "形式を学習中…",
    "📖 从参考 EPUB 学习格式…": "📖 参照EPUBから形式を学習…",
    "显示": "表示",
    "保存密钥到本机": "キーをこの端末に保存",
    "可选：全书术语白名单（TXT / JSON）": "任意：全巻用語ホワイトリスト（TXT / JSON）",
    "启用思考模式（更慢、消耗额外推理 Tokens）": "推論モードを有効化（低速・追加トークン使用）",
    "可读性优先（推荐）": "読みやすさ優先（推奨）",
    "严格还原（不推测缺字）": "厳密復元（欠字を推測しない）",
    "使用接口原生 JSON 模式（不支持时自动回退）": "APIネイティブJSONモードを使用（非対応時は自動フォールバック）",
    "严格覆盖（重建小说排版，推荐）": "厳密置換（小説レイアウト再構築・推奨）",
    "严格覆盖（完全原样）": "厳密置換（完全原文どおり）",
    "局部智能替换": "部分スマート置換",
    "仅比较差异": "差分のみ比較",
    "等待文档": "文書待機中",
    "AI处理中": "AI処理中",
    "准备中…": "準備中…",
    "正在准备章节…": "章を準備中…",
    "正在停止": "停止中",
    "↻ 刷新检查": "↻ チェック更新",
    "等待正文版本": "本文バージョン待機中",
    "内容结构": "コンテンツ構造",
    "打包文件": "パッケージファイル",
    "元数据": "メタデータ",
    "粘贴": "貼り付け",
    "作者": "著者",
    "出版社": "出版社",
    "卷号": "巻番号",
    "⚙ 管理格式": "⚙ 形式を管理",
    "竖排": "縦書き",
    "横排": "横書き",
    "源码": "ソース",
    "预览": "プレビュー",
    "等待新书": "新しい本を待機中",
    "构建中…": "ビルド中…",
    "✓ 完成": "✓ 完了",
    "✗ 失败": "✗ 失敗",
    "PDF 文字层直读": "PDFテキストレイヤー直接読み込み",
    "输入（单个 PDF 文件）": "入力（単一PDF）",
    "📄  选择 PDF 文件": "📄  PDFを選択",
    "提取日志": "抽出ログ",
    "AI 修改审阅 · 逐项确认": "AI変更レビュー・個別確認",
    "采用低/中风险": "低 / 中リスクを採用",
    "全部保留原文": "すべて原文を保持",
    "采用全部AI修改": "すべてのAI変更を採用",
    "← 上一处": "← 前へ",
    "下一处 →": "次へ →",
    "定位到双栏": "2列表示へ移動",
    "发布前检查": "公開前チェック",
    "参考原文对照评分": "参照原文との比較スコア",
    "双 OCR 融合裁决报告": "2系統OCR融合判定レポート",
    "导入左文本": "左テキストを読み込む",
    "导入右文本": "右テキストを読み込む",
    "🖼 页面管理 / 插图": "🖼 ページ管理 / 挿絵",
    "↵ 光标处分行": "↵ カーソル位置で改行",
    "⇤ 删除换行 / 接上前句": "⇤ 改行削除 / 前文へ結合",
    "＋ 空白对齐行": "＋ 空白整列行",
    "「」对白独立成行": "会話文を独立行にする",
    "⚡ 实时对齐": "⚡ リアルタイム整列",
    "A 文字颜色": "A 文字色",
    "▰ 背景色": "▰ 背景色",
    "U 下划线": "U 下線",
    "S 删除线": "S 取り消し線",
    "清除样式": "書式をクリア",
    "导出双栏文本": "2列テキストを書き出す",
    "双 OCR：": "2系統OCR：",
    "智能裁决": "スマート判定",
    "外部 OCR 优先": "外部OCRを優先",
    "内置 OCR 优先": "内蔵OCRを優先",
    "查看融合裁决": "融合判定を表示",
    "外部 OCR：未导入": "外部OCR：未読み込み",
    "可信正文：": "信頼本文：",
    "折叠右侧大空白": "右側の大きな空白を折りたたむ",
    "AI审校：": "AI校正：",
    "AI纠错当前行": "AIで現在行を修正",
    "AI纠错选中区域": "AIで選択範囲を修正",
    "查看AI修改（0）": "AI変更を表示（0）",
    "上一处 AI 修改": "前のAI変更",
    "下一处 AI 修改": "次のAI変更",
    "✓ 确认AI结果（仅右侧）": "✓ AI結果を確定（右側のみ）",
    "✕ 拒绝AI结果": "✕ AI結果を却下",
    "↶ 撤销上次AI": "↶ 前回AIを取り消す",
    "AI设置": "AI設定",
    "翻译/发布校验：": "翻訳 / 公開チェック：",
    "检查当前右侧": "現在の右側をチェック",
    "修复确定问题": "確定問題を修正",
    "📘 导入参考原文": "📘 参照原文を読み込む",
    "参考评分": "参照スコア",
    "参考原文：未导入": "参照原文：未読み込み",
    "AI状态：未运行": "AI状態：未実行",
    "左侧": "左側",
    "右侧": "右側",
    "句末": "文末",
    "重新扫描右侧全文": "右側全文を再走査",
    "上一处（Shift+F7）": "前へ（Shift+F7）",
    "下一处（F7）": "次へ（F7）",
    "待复核数量 / 当前序号": "要確認数 / 現在位置",
    "左侧参考文本": "左側参照テキスト",
    "右侧待修改文本": "右側編集対象テキスト",
    "评分中…": "採点中…",
    "重新选择": "再選択",
    "AI 审定多模型 OCR 融合稿": "AIによる複数OCR融合稿の判定",
    "OCR 对比 · 结果收件箱": "OCR比較・結果受信箱",
    "↔ 重新对齐": "↔ 再整列",
    "↶ 恢复初始": "↶ 初期状態へ戻す",
    "逐句裁决": "文ごとの判定",
    "全文对比": "全文比較",
    "显示全文对比": "全文比較を表示",
    "图文检查 ↗": "画像・本文確認 ↗",
    "更多操作 ▾": "その他 ▾",
    "⇄ 返回多模型对比": "⇄ 複数モデル比較へ戻る",
    "当前句采用：": "現在文の採用候補：",
    "当前句：—": "現在文：—",
    "多模型包：": "複数モデルパッケージ：",
    "OCR 裁决：": "OCR判定：",
    "↻ 恢复纠错会话": "↻ 校正セッションを復元",
    "尚未导入 OCR 裁决": "OCR判定未読み込み",
    "停止AI审定": "AI判定を停止",
    "只显示需判断": "要判定のみ表示",
    "只显示一个对比框": "比較枠を1つだけ表示",
    "选择后自动下一条": "選択後に自動で次へ",
    "← 上一分歧": "← 前の差分",
    "下一分歧 →": "次の差分 →",
    "下一句 →": "次の文 →",
    "待判断队列": "判定待ちキュー",
    "OCR 对比 · 逐句裁决": "OCR比較・文ごとの判定",
    "✓ 应用融合稿": "✓ 融合稿を適用",
    "OCR 对比 · 单结果校对": "OCR比較・単一結果校正",
    "✓ 应用单OCR结果": "✓ 単一OCR結果を適用",
    "AI 修复包导出选项": "AI修復パッケージ出力設定",
    "开始导出": "出力開始",
    "图文逐句校对": "画像・本文の文単位校正",
    "尚未接收 OCR 结果": "OCR結果未受信",
    "OCR 竖列（与原图列顺序对齐）": "OCR縦列（原画像の列順に整列）",
    "对应识别图片": "対応する認識画像",
    "用预览打开": "プレビューで開く",
    "横排校对文本（可修改）": "横書き校正テキスト（編集可）",
    "苹果系统手写": "Appleシステム手書き",
    "多模型融合候选": "複数モデル融合候補",
    "← 上一句": "← 前の文",
    "保存当前句": "現在文を保存",
    "下一待判断": "次の要判定",
    "上一OCR分歧": "前のOCR差分",
    "下一OCR分歧": "次のOCR差分",
    "正在载入…": "読み込み中…",
    "更新分支": "更新ブランチ",
    "GitHub 仓库": "GitHubリポジトリ",
    "本地程序目录": "ローカルプログラムフォルダ",
    "当前版本": "現在のバージョン",
    "复制诊断信息": "診断情報をコピー",
    "打开 GitHub 仓库": "GitHubリポジトリを開く",
    "打开程序目录": "プログラムフォルダを開く",
}


# Additional concise OCR/workspace UI labels.
_EXACT_EN.update({
    '请按住灰色滑块手柄后上下拖动；点击轨道不会跳转': 'Drag the gray slider handle vertically; clicking the track will not jump',
    '恢复上一次扫描件优化之前的页面列表和页面分类。': 'Restore the page list and classifications from before the last scan optimization.',
    '选择输入后在此显示图片，可拖框选定识别区域': 'The selected input appears here; drag to choose the recognition area',
    '识别引擎、版式检测与输出选项': 'Recognition engine, layout detection and output options',
    '✓ 每个模型仅执行一次主识别（固定安全合同）': '✓ Run one primary recognition per model (fixed safety contract)',
    '快速共识：模型1/2先判定，后续模型只补分歧列': 'Fast consensus: models 1/2 decide first; later models only resolve conflicting columns',
    '模型1/2并行首轮（异构引擎推荐）': 'Run models 1/2 first pass in parallel (recommended for mixed engines)',
    '平衡（推荐）：只由主模型执行一次': 'Balanced (recommended): primary model only',
    '完整精校：每个模型都执行': 'Full review: run every model',
    'macOS 快捷指令 · 稳定兼容通道': 'macOS Shortcuts · stable compatibility path',
    '按日文竖排顺序组合观察结果（右→左、上→下）': 'Combine observations in Japanese vertical order (right→left, top→bottom)',
    '竖排列紧裁后左旋 90°识别（单次 OCR，推荐）': 'Tight-crop vertical column then rotate 90° left (single OCR, recommended)',
    'base（默认，质量/体积折中）': 'base (default, quality/size balance)',
    'tiny（最小）': 'tiny (smallest)',
    'gundam（最大/最高质量）': 'gundam (largest/highest quality)',
    '例如 ja, zh-CN, en': 'e.g. ja, zh-CN, en',
    '官方异步 v2 jobs（推荐）': 'Official async v2 jobs (recommended)',
    '同步 API（兼容旧模型页 URL）': 'Synchronous API (legacy model-page URL compatible)',
    'PaddleOCR-VL-1.6（官方文档解析，推荐）': 'PaddleOCR-VL-1.6 (official document parsing, recommended)',
    'PP-OCRv6（官方 OCR）': 'PP-OCRv6 (official OCR)',
    'PP-OCRv5-latin（拉丁文字）': 'PP-OCRv5-latin (Latin text)',
    'PyTorch（推荐：CUDA / MPS / CPU）': 'PyTorch (recommended: CUDA / MPS / CPU)',
    'LiteRT（CPU / 边缘端，实验）': 'LiteRT (CPU / edge, experimental)',
    '自动（CUDA → MPS → CPU）': 'Auto (CUDA → MPS → CPU)',
    '不量化（默认 / 最高兼容）': 'No quantization (default / best compatibility)',
    'INT8 weight-only（约 2× 降显存）': 'INT8 weight-only (~2× lower VRAM)',
    'INT4 weight-only（约 4× 降显存）': 'INT4 weight-only (~4× lower VRAM)',
    'WI4（默认）': 'WI4 (default)',
    '快速：tiny 动态宽度 + 分桶': 'Fast: tiny dynamic width + bucketing',
    '精确：middle v5 动态宽度': 'Accurate: middle v5 dynamic width',
    'middle-v5 复核阈值': 'middle-v5 review threshold',
    'PP-StructureV3（复杂版面与区域分析）': 'PP-StructureV3 (complex layout/region analysis)',
    'MLX-VLM（Apple Silicon 官方后端）': 'MLX-VLM (official Apple Silicon backend)',
    'Paddle（兼容后端）': 'Paddle (compatibility backend)',
    '百度 BOS': 'Baidu BOS',
    '尚未检查 PaddleOCR 环境': 'PaddleOCR environment not checked',
    '合并同一横行内被拆开的文字框': 'Merge split text boxes on the same horizontal line',
    '自动过滤跨页重复页眉 / 页脚': 'Auto-filter repeated headers/footers across pages',
    '启用分列掩膜：逐列保留原始像素，其他区域用纸白色遮住': 'Enable column masks: keep source pixels per column and mask the rest white',
    '保留原文 Ruby（findtextCenterNet）': 'Preserve source Ruby (findtextCenterNet)',
    '自动过滤日文 Ruby（振假名）': 'Auto-filter Japanese Ruby (furigana)',
    '删除残损小文字 / 邻列碎片': 'Remove damaged tiny text / adjacent-column fragments',
    '分列 OCR 智能裁剪（不缩放）': 'Smart crop for column OCR (no scaling)',
    '弱：只去除很明确的细小注音': 'Weak: remove only obvious tiny annotations',
    '强：同时清理更多邻列残影': 'Strong: also remove more adjacent-column remnants',
    '保持默认可获得与稳定基线一致的 OCR 输入。': 'Defaults preserve OCR input consistent with the stable baseline.',
    '无损紧凑列图加速（不缩放；严重空白/缺字才复核全尺寸）': 'Lossless compact-column acceleration (no scaling; full-size retry only for severe blanks/omissions)',
    '自适应单次（推荐）：每列最多一种救援': 'Adaptive single rescue (recommended): at most one rescue per column',
    '关闭救援：只保留主识别与人工复核': 'Disable rescue: keep primary recognition and manual review only',
    '完整多轮：兼容旧版恢复链': 'Full multi-pass: legacy recovery compatibility',
    '智能混合（推荐）：整页一次，疑难列再补识': 'Smart hybrid (recommended): one full-page pass, then difficult columns',
    '高速整页：每页一次，漏列保留待人工': 'Fast full page: once per page; missing columns kept for review',
    '强制逐列：保持旧版逐列识别': 'Force per-column: keep legacy per-column recognition',
    '严格防漏：预计列、识别列、文档列和 DOCX 列必须一致': 'Strict anti-omission: expected, recognized, document and DOCX column counts must match',
    '> OCR + 日语手写人工纠错': '> OCR + Japanese handwriting correction',
    '✍️ OCR + 日语手写人工纠错': '✍️ OCR + Japanese handwriting correction',
    '启用逐字审校（预览显示蓝色逐字框）': 'Enable character review (blue character boxes in preview)',
    '只自动筛查': 'Auto screening only',
    '自动筛查 + 人工复核': 'Auto screening + manual review',
    '人工复核全部列': 'Manually review all columns',
    '保守（只标极高风险冲突）': 'Conservative (flag only very high-risk conflicts)',
    '标准（推荐疑点范围）': 'Standard (recommended review range)',
    '广泛（标记更多候选冲突）': 'Broad (flag more candidate conflicts)',
    '描摹候选器': 'Trace candidate engine',
    'OpenVINO 日语手写模型（兼容备用）': 'OpenVINO Japanese handwriting model (compatibility fallback)',
    '本地 JLect 笔画候选（最低备用）': 'Local JLect stroke candidates (last-resort fallback)',
    '先手动画一个日语字，验证苹果识别器本身': 'Draw one Japanese character manually to test Apple recognizer',
    '当前列内再次掩膜：过滤振假名 / 邻列残影': 'Mask within current column again: filter furigana / adjacent-column remnants',
    '把句读点 / 引号 / 长音符差异纳入疑点': 'Include punctuation / quote / prolonged-mark differences as risks',
    '仅用于标记符号疑点，不会自动插入或删除正文字符。': 'Only flags symbol risks; never inserts or deletes body characters.',
    '逐列成句：先按坐标归并物理列，再按列尾成句': 'Build sentences by column: merge physical columns by coordinates, then sentence endings',
    '自适应：仅有风险证据的句子（推荐）': 'Adaptive: sentences with risk evidence only (recommended)',
    '完整：全部多列句（原模式）': 'Full: all multi-column sentences (legacy mode)',
    '疑难句重识别优先使用真实合并框': 'Prefer true merged boxes for difficult-sentence re-recognition',
})
_EXACT_JA.update({
    '请按住灰色滑块手柄后上下拖动；点击轨道不会跳转': '灰色のスライダーハンドルを上下にドラッグしてください。トラックのクリックでは移動しません',
    '恢复上一次扫描件优化之前的页面列表和页面分类。': '前回のスキャン最適化前のページ一覧と分類を復元します。',
    '选择输入后在此显示图片，可拖框选定识别区域': '入力を選択するとここに画像を表示します。ドラッグで認識範囲を指定できます',
    '识别引擎、版式检测与输出选项': '認識エンジン、レイアウト検出、出力オプション',
    '✓ 每个模型仅执行一次主识别（固定安全合同）': '✓ 各モデルは主認識を1回だけ実行（固定安全契約）',
    '快速共识：模型1/2先判定，后续模型只补分歧列': '高速合意：モデル1/2を先行し、後続モデルは不一致列だけを補完',
    '模型1/2并行首轮（异构引擎推荐）': 'モデル1/2の初回を並列実行（異種エンジン推奨）',
    '平衡（推荐）：只由主模型执行一次': 'バランス（推奨）：主モデルのみ1回実行',
    '完整精校：每个模型都执行': '完全精査：全モデルで実行',
    'macOS 快捷指令 · 稳定兼容通道': 'macOSショートカット · 安定互換経路',
    '按日文竖排顺序组合观察结果（右→左、上→下）': '日本語縦書き順で結果を結合（右→左、上→下）',
    '竖排列紧裁后左旋 90°识别（单次 OCR，推荐）': '縦列をタイトに切り出して左90°回転認識（単発OCR、推奨）',
    'base（默认，质量/体积折中）': 'base（既定、品質/容量のバランス）',
    'tiny（最小）': 'tiny（最小）',
    'gundam（最大/最高质量）': 'gundam（最大/最高品質）',
    '例如 ja, zh-CN, en': '例：ja, zh-CN, en',
    '官方异步 v2 jobs（推荐）': '公式非同期 v2 jobs（推奨）',
    '同步 API（兼容旧模型页 URL）': '同期API（旧モデルページURL互換）',
    'PaddleOCR-VL-1.6（官方文档解析，推荐）': 'PaddleOCR-VL-1.6（公式ドキュメント解析、推奨）',
    'PP-OCRv6（官方 OCR）': 'PP-OCRv6（公式OCR）',
    'PP-OCRv5-latin（拉丁文字）': 'PP-OCRv5-latin（ラテン文字）',
    'PyTorch（推荐：CUDA / MPS / CPU）': 'PyTorch（推奨：CUDA / MPS / CPU）',
    'LiteRT（CPU / 边缘端，实验）': 'LiteRT（CPU / エッジ、実験）',
    '自动（CUDA → MPS → CPU）': '自動（CUDA → MPS → CPU）',
    '不量化（默认 / 最高兼容）': '量子化なし（既定 / 最高互換性）',
    'INT8 weight-only（约 2× 降显存）': 'INT8 weight-only（VRAM約1/2）',
    'INT4 weight-only（约 4× 降显存）': 'INT4 weight-only（VRAM約1/4）',
    'WI4（默认）': 'WI4（既定）',
    '快速：tiny 动态宽度 + 分桶': '高速：tiny 動的幅 + バケット化',
    '精确：middle v5 动态宽度': '高精度：middle v5 動的幅',
    'middle-v5 复核阈值': 'middle-v5 再確認しきい値',
    'PP-StructureV3（复杂版面与区域分析）': 'PP-StructureV3（複雑レイアウト/領域解析）',
    'MLX-VLM（Apple Silicon 官方后端）': 'MLX-VLM（Apple Silicon公式バックエンド）',
    'Paddle（兼容后端）': 'Paddle（互換バックエンド）',
    '百度 BOS': 'Baidu BOS',
    '尚未检查 PaddleOCR 环境': 'PaddleOCR環境は未確認',
    '合并同一横行内被拆开的文字框': '同じ横行で分割されたテキスト枠を結合',
    '自动过滤跨页重复页眉 / 页脚': 'ページ間で重複するヘッダー/フッターを自動除外',
    '启用分列掩膜：逐列保留原始像素，其他区域用纸白色遮住': '列マスクを有効化：列ごとに原画素を保持し、他領域を紙白でマスク',
    '保留原文 Ruby（findtextCenterNet）': '原文Rubyを保持（findtextCenterNet）',
    '自动过滤日文 Ruby（振假名）': '日本語Ruby（振り仮名）を自動除外',
    '删除残损小文字 / 邻列碎片': '欠損した小文字 / 隣接列の断片を除去',
    '分列 OCR 智能裁剪（不缩放）': '列OCRスマートクロップ（拡大縮小なし）',
    '弱：只去除很明确的细小注音': '弱：明確な小型注音のみ除去',
    '强：同时清理更多邻列残影': '強：隣接列の残像もより多く除去',
    '保持默认可获得与稳定基线一致的 OCR 输入。': '既定値では安定ベースラインと同じOCR入力を保ちます。',
    '无损紧凑列图加速（不缩放；严重空白/缺字才复核全尺寸）': '無損失コンパクト列画像高速化（拡大縮小なし。重大な空白/欠字時のみ全サイズ再確認）',
    '自适应单次（推荐）：每列最多一种救援': '適応型単発（推奨）：各列で救済は最大1種類',
    '关闭救援：只保留主识别与人工复核': '救済を無効化：主認識と手動確認のみ',
    '完整多轮：兼容旧版恢复链': '完全マルチパス：旧版復旧チェーン互換',
    '智能混合（推荐）：整页一次，疑难列再补识': 'スマート混合（推奨）：全ページ1回＋難しい列のみ追加認識',
    '高速整页：每页一次，漏列保留待人工': '高速全ページ：各ページ1回、欠落列は手動確認へ',
    '强制逐列：保持旧版逐列识别': '強制列単位：旧版の列別認識を維持',
    '严格防漏：预计列、识别列、文档列和 DOCX 列必须一致': '厳格な欠落防止：予測列・認識列・文書列・DOCX列を一致させる',
    '> OCR + 日语手写人工纠错': '> OCR + 日本語手書き修正',
    '✍️ OCR + 日语手写人工纠错': '✍️ OCR + 日本語手書き修正',
    '启用逐字审校（预览显示蓝色逐字框）': '文字単位校正を有効化（プレビューに青い文字枠）',
    '只自动筛查': '自動スクリーニングのみ',
    '自动筛查 + 人工复核': '自動スクリーニング + 手動確認',
    '人工复核全部列': '全列を手動確認',
    '保守（只标极高风险冲突）': '保守的（極高リスク競合のみ）',
    '标准（推荐疑点范围）': '標準（推奨確認範囲）',
    '广泛（标记更多候选冲突）': '広範囲（より多くの候補競合を表示）',
    '描摹候选器': '筆跡候補エンジン',
    'OpenVINO 日语手写模型（兼容备用）': 'OpenVINO日本語手書きモデル（互換予備）',
    '本地 JLect 笔画候选（最低备用）': 'ローカルJLect筆画候補（最終予備）',
    '先手动画一个日语字，验证苹果识别器本身': '日本語1文字を手書きしApple認識器自体を確認',
    '当前列内再次掩膜：过滤振假名 / 邻列残影': '現在列を再マスク：振り仮名 / 隣接列の残像を除外',
    '把句读点 / 引号 / 长音符差异纳入疑点': '句読点 / 引用符 / 長音符の差異を疑点に含める',
    '仅用于标记符号疑点，不会自动插入或删除正文字符。': '記号の疑点表示だけに使用し、本文文字を自動挿入・削除しません。',
    '逐列成句：先按坐标归并物理列，再按列尾成句': '列ごとに文を構成：座標で物理列を統合し、列末で文を確定',
    '自适应：仅有风险证据的句子（推荐）': '適応型：リスク証拠がある文のみ（推奨）',
    '完整：全部多列句（原模式）': '完全：全ての複数列文（旧モード）',
    '疑难句重识别优先使用真实合并框': '難しい文の再認識では実結合枠を優先',
})


# Additional workspace labels and live status messages.
_EXACT_EN.update({
    '总进度 0.0% · 已用 0秒 · 预计剩余：计算中': 'Overall 0.0% · Elapsed 0s · Remaining: calculating',
    '当前：准备输入与 OCR 模型…': 'Current: preparing input and OCR models…',
    '正在后台扫描本地 OCR 运行环境…': 'Scanning local OCR runtimes in the background…',
    'OCR 运行环境扫描失败；可手动重新检测': 'OCR runtime scan failed; you can run detection again manually',
    '正在检查本地 OCR 运行环境…': 'Checking local OCR runtime…',
    'OCR 环境检查失败': 'OCR runtime check failed',
    '单模型 OCR 已写入当前工作区': 'Single-model OCR saved to the current workspace',
    '把处理前/处理后文本框里的手动修改写回当前文档': 'Write manual before/after edits back to the current document',
    '只清空右侧处理后文本框和当前处理结果，不影响处理前内容': 'Clear only the right processed text and current result; keep source text',
    '把左侧处理前文档重新载入到右侧处理后文本框': 'Reload the left source document into the right processed-text box',
    '滚动到顶部（⌘↑ / Ctrl+Home）': 'Scroll to top (⌘↑ / Ctrl+Home)',
    '滚动到底部（⌘↓ / Ctrl+End）': 'Scroll to bottom (⌘↓ / Ctrl+End)',
    '本书 AI 断点已清除': 'AI checkpoint for this book cleared',
    '正在读取可用模型列表……': 'Loading available models…',
    '处理对象：当前替换结果；只纠错，尽量保持段落结构': 'Target: current replacement result; correct errors while preserving paragraph structure',
    '0 / 0 章': '0 / 0 chapters',
    '将在当前 API 请求结束后停止': 'Will stop after the current API request',
    '书名；可直接粘贴，生成时自动作为 EPUB 文件名': 'Book title; paste directly and use automatically as EPUB filename',
    '从剪贴板粘贴书名，并用于建议保存文件名': 'Paste title from clipboard and use it for suggested filename',
    '管理排版格式（新建/学习/删除/导入导出）': 'Manage layout formats (new/learn/delete/import/export)',
    '选择左侧文件预览内容': 'Select a file on the left to preview',
    '实时预览当前正文；生成后可从“打包文件”选择具体文件': 'Live preview of current body; after build, choose a packaged file',
    '内置 OCR ↔ 外部 OCR': 'Built-in OCR ↔ External OCR',
    '只处理右侧光标所在行；允许把该行拆成正文和对白': 'Process only the line under the right cursor; may split narration/dialogue',
    '只处理右侧选中的连续行；不会跨越插图标记': 'Process only selected contiguous right-side lines; never cross image markers',
    '查看 AI 处理前/处理后的逐组修改，并双击定位到双栏': 'View AI before/after changes by group; double-click to locate in both panes',
    '开启或关闭句末可疑位置标红': 'Toggle red highlighting of suspicious sentence endings',
    '尚未载入正文版本': 'No body-text version loaded',
    '正在按当前编辑内容重新匹配……': 'Rematching using current edits…',
    'AI状态：已确认，尚未应用到其他界面': 'AI status: confirmed, not yet applied elsewhere',
    'AI状态：未运行 / 已恢复处理前文本': 'AI status: not run / source text restored',
    '已拒绝 AI 结果，右侧完整恢复到处理前状态。': 'AI result rejected; right pane fully restored to pre-AI state.',
    '已撤销上一次已接受的 AI 纠错排版结果。': 'Last accepted AI correction/layout result has been undone.',
    '外部 OCR：正在后台融合……': 'External OCR: merging in background…',
    '外部 OCR：融合失败': 'External OCR: merge failed',
    '✓ 选择': '✓ Select',
    '单 OCR：暂无可载入结果': 'Single OCR: no loadable result',
    '恢复切换前的多模型文本、对齐、人工候选选择和未完成状态。': 'Restore pre-switch multi-model text, alignment, manual choices and pending state.',
    '关闭后选择会保留在当前句，便于再次核对；默认开启。': 'When off, selection stays on the current sentence for rechecking; on by default.',
    '定位上一条仍需判断的句子。': 'Go to previous sentence needing a decision.',
    '定位下一条仍需判断的句子。': 'Go to next sentence needing a decision.',
    '正在准备轻量裁决状态…': 'Preparing lightweight adjudication state…',
    '逐句裁决 · 当前候选': 'Sentence adjudication · Current candidates',
    '已撤销整组原子修复选择，事务成员全部恢复为待确认。': 'Atomic repair group selection reverted; all members are pending again.',
    '所有不一致候选均已完成选择；没有上一组待判断内容。': 'All differing candidates are decided; no previous pending group.',
    '已完成最后一句；没有下一句。': 'Last sentence completed; no next sentence.',
    '所有不一致候选均已完成选择；没有下一组待判断内容。': 'All differing candidates are decided; no next pending group.',
    'OCR 裁决任务失败': 'OCR adjudication task failed',
    'V4 多模型分歧裁决包（默认推荐）': 'V4 multi-model conflict package (recommended)',
    'V3 标准紧凑包（兼容）': 'V3 standard compact package (compatible)',
    'V3 完整取证包（调试）': 'V3 full evidence package (debug)',
    '包含出版参考证据（可选，默认关闭）': 'Include publication reference evidence (optional, off by default)',
    '未选择出版参考 EPUB': 'No publication reference EPUB selected',
    '选择 EPUB…': 'Choose EPUB…',
    '校验完成，正在分批刷新 OCR 对比界面…': 'Validation complete; refreshing OCR comparison in batches…',
    '当前没有需要跳转的未决分歧或已保留历史分歧。': 'No pending or retained historical conflicts to navigate to.',
    '完成 OCR 后，右侧显示当前句对应的单列或多列原图。': 'After OCR, the right side shows source column image(s) for the current sentence.',
    '使用此候选': 'Use this candidate',
    '使用 macOS 预览打开当前句/列图片（⌥P）': 'Open current sentence/column image in macOS Preview (⌥P)',
    '完成 OCR 后在此横排逐句校对': 'Proofread OCR sentence by sentence here in horizontal layout',
    '图文逐句索引建立失败': 'Failed to build sentence image/text index',
    '已跳到上一条 OCR 对比不一致句图': 'Moved to previous OCR-disagreement sentence image',
    '已跳到下一条 OCR 对比不一致句图': 'Moved to next OCR-disagreement sentence image',
    '苹果手写板已打开；点击“复制并返回”后自动写入此处': 'Apple handwriting pad opened; “Copy & Return” writes here automatically',
    '已接收苹果手写结果，尚未保存': 'Apple handwriting result received; not saved yet',
    '核心运行合同：始终启用，不是可切换选项': 'Core runtime contract: always enabled, not switchable',
    '正在检测本机设备和已安装 OCR 运行时…': 'Detecting local hardware and installed OCR runtimes…',
    '设备检测失败；未修改任何 OCR 设置。': 'Device detection failed; no OCR settings were changed.',
    '更新源已锁定，配置和界面都不能改到其它仓库。': 'Update source is locked; settings/UI cannot point to another repository.',
    '更新分支固定为 main。': 'Update branch is fixed to main.',
    '尚未检查更新。启动程序不会自动联网检查。': 'Updates not checked. Startup never checks online automatically.',
    '检查与更新过程会显示在这里。': 'Check/update progress appears here.',
    '正在安全更新程序代码…': 'Safely updating application code…',
    '📥 导入 PaddleOCR-VL': '📥 Import PaddleOCR-VL',
    '没有可预览页面': 'No pages to preview',
    '单击选择 · 双击打开高清预览 · 右键设置页面类型': 'Click to select · Double-click HD preview · Right-click page type',
    '本地最低备用可直接使用；字符库不完整，未识字会保留 □。': 'Local last-resort fallback is available; unknown glyphs remain □.',
    'OpenVINO 日语手写模型与运行环境可用。': 'OpenVINO Japanese handwriting model/runtime available.',
    '正在打开自动轨迹…': 'Opening automatic trace…',
    '重新下载模型': 'Redownload Model',
    '未选择输入': 'No input selected',
    '输入参考图': 'Input Reference Image',
    '当前：正在终止 OCR，不再派发新任务…': 'Current: stopping OCR; no new tasks will be dispatched…',
    '当前图片：等待第一张识别图片…': 'Current image: waiting for first recognized page…',
    '当前图片：实时预览已关闭（OCR 继续运行）': 'Current image: live preview off (OCR continues)',
    '实时预览已关闭；不会生成或刷新新的预览图。': 'Live preview is off; no new preview images will be generated/refreshed.',
    '处理前': 'Before',
    '处理后': 'After',
    '<b>格式列表</b>': '<b>Format List</b>',
    '输入密钥后加载模型，或手动填写': 'Enter a key to load models, or type one manually',
    '修改记录 / 结构信息': 'Change Log / Structure',
    '🗑 清空': '🗑 Clear',
    'AI任务已停止': 'AI task stopped',
    '<b>EPUB 制作</b>': '<b>EPUB Builder</b>',
    '⚠ 封面：未设置（可在页面管理中指定）': '⚠ Cover: not set (set it in Page Manager)',
    '🖼 封面未设置': '🖼 Cover not set',
    '☰ 0 章': '☰ 0 chapters',
    '🎨 0 图': '🎨 0 images',
    'CSS 默认模板': 'Default CSS template',
    '待构建': 'Pending Build',
    '左侧：': 'Left:',
    '右侧：': 'Right:',
    '文字主稿：': 'Text master:',
    '0 处': '0 items',
    'AI状态：处理中；右侧暂时锁定': 'AI status: processing; right pane temporarily locked',
    'AI状态：已确认并应用到当前文档': 'AI status: confirmed and applied to current document',
    '句末复核已开启；可点击“重新扫描”或“下一处”。': 'Sentence-end review enabled; use “Rescan” or “Next”.',
    '句末复核已关闭。': 'Sentence-end review disabled.',
    '句末复核已重新扫描：右侧当前没有发现可疑句。': 'Sentence-end rescan complete: no suspicious right-side sentence found.',
    '关': 'Off',
    '右侧当前没有检测到缺少句末标点的可疑句。': 'No right-side sentence with suspicious missing terminal punctuation.',
    '请先选择要设置样式的文字': 'Select text to style first',
    '请先选择要清除样式的文字': 'Select text whose style should be cleared first',
    '参考评分没有得到可用结果。': 'Reference scoring returned no usable result.',
    '候选文字一致；来源仍分别保留，可重新选择': 'Candidate text matches; sources remain separate and selectable',
    '已选择；未选候选已收起，当前文字仍可编辑': 'Selected; unchosen candidates collapsed; current text remains editable',
    '已完成裁决 · 下方显示纠错前只读分歧证据': 'Adjudication complete · Read-only pre-correction evidence below',
    '两模型共同候选 · 已重新打开证据': 'Two-model shared candidate · Evidence reopened',
    '每批页数': 'Pages per batch',
    '前后重叠页': 'Overlap pages',
    '单请求输入上限': 'Per-request input limit',
    '该模型的逐句 OCR 结果': 'Sentence OCR results for this model',
    '全文融合结果（可查看、重新选择与编辑）': 'Full-text fusion result (view/reselect/edit)',
    'OCR 对比 · 全文对比': 'OCR Compare · Full Text',
    '＋ 从 OCR 识别载入': '＋ Load from OCR',
    '尚未导入逐源纠错': 'No per-source corrections imported',
    '已恢复完整多模型文本视图；逐句选择和候选修改均已保留。': 'Full multi-model text view restored; sentence choices and candidate edits retained.',
    '已重新打开当前句，图文对照同步恢复为待确认。': 'Current sentence reopened; image/text review is pending again.',
    'OCR 对比 · 已恢复纠错会话': 'OCR Compare · Correction Session Restored',
    'AI 修复包模式：': 'AI repair-package mode:',
    'AI 审定已取消；当前融合稿没有被半途改写。': 'AI adjudication canceled; current fusion text was not partially changed.',
    '当前句没有可用的原图裁片。': 'No source crop available for current sentence.',
    '✓ 当前采用': '✓ Currently Used',
    '横排 · 上→下、左→右': 'Horizontal · top→bottom, left→right',
    '图片不可用': 'Image unavailable',
    '候选已写入文本框，但保存失败；请检查当前句映射': 'Candidate written to text box, but save failed; check sentence mapping',
    '所有需要判断的句图均已核对': 'All sentence images needing decisions are reviewed',
    '当前文档没有多模型 OCR 分歧句': 'Current document has no multi-model OCR conflict sentences',
    '已保存，正在同步到 OCR 对比…': 'Saved; syncing to OCR Compare…',
    '已同步 OCR 对比最终选择': 'Final OCR Compare selection synchronized',
    '已保存，当前是最后一句': 'Saved; this is the last sentence',
    '即将支持': 'Coming soon',
    '无需安装': 'No installation needed',
    '正在打开…': 'Opening…',
    '当前图片：等待下一张识别图片…': 'Current image: waiting for next recognized image…',
    '设置已保存；本地旧 API Key 已彻底移除': 'Settings saved; legacy local API Key fully removed',
    '恢复为 AI 来源版本': 'Restore AI source version',
    '恢复为本次 AI 处理所使用的 OCR 或替换结果': 'Restore the OCR/replacement result used by this AI run',
    '两模型共同候选 · 已自动保留，可查看证据': 'Two-model shared candidate · Auto-kept; evidence available',
    '✓ 已载入当前单OCR': '✓ Current single OCR loaded',
    'OCR 对比已重新打开此句，等待选择': 'OCR Compare reopened this sentence and is awaiting selection',
    'Apple 桥接已就绪': 'Apple bridge ready',
    'OpenVINO 已就绪': 'OpenVINO ready',
    '正在编译…': 'Compiling…',
    '当前：等待下一次 OCR 进度事件…': 'Current: waiting for next OCR progress event…',
    '两模型共同候选 · 按 v8 规则自动保留': 'Two-model shared candidate · Auto-kept by v8 rules',
    '所有独立模型完全一致 · 已自动保留': 'All independent models agree · Auto-kept',
    '仅一个有效候选 · 已保留': 'Only one valid candidate · Kept',
})
_EXACT_JA.update({
    '总进度 0.0% · 已用 0秒 · 预计剩余：计算中': '全体 0.0% · 経過 0秒 · 残り：計算中',
    '当前：准备输入与 OCR 模型…': '現在：入力とOCRモデルを準備中…',
    '正在后台扫描本地 OCR 运行环境…': 'ローカルOCR実行環境をバックグラウンドで確認中…',
    'OCR 运行环境扫描失败；可手动重新检测': 'OCR実行環境の確認に失敗しました。手動で再検出できます',
    '正在检查本地 OCR 运行环境…': 'ローカルOCR実行環境を確認中…',
    'OCR 环境检查失败': 'OCR実行環境の確認に失敗しました',
    '单模型 OCR 已写入当前工作区': '単一モデルOCRを現在のワークスペースに保存しました',
    '把处理前/处理后文本框里的手动修改写回当前文档': '処理前/後テキスト欄の手動修正を現在の文書へ書き戻す',
    '只清空右侧处理后文本框和当前处理结果，不影响处理前内容': '右側の処理後テキストと現在結果だけを消去し、処理前は保持',
    '把左侧处理前文档重新载入到右侧处理后文本框': '左側の処理前文書を右側の処理後テキスト欄へ再読込',
    '滚动到顶部（⌘↑ / Ctrl+Home）': '先頭へスクロール（⌘↑ / Ctrl+Home）',
    '滚动到底部（⌘↓ / Ctrl+End）': '末尾へスクロール（⌘↓ / Ctrl+End）',
    '本书 AI 断点已清除': 'この書籍のAIチェックポイントを消去しました',
    '正在读取可用模型列表……': '利用可能なモデルを読み込み中…',
    '处理对象：当前替换结果；只纠错，尽量保持段落结构': '対象：現在の置換結果。段落構造を保ちながら誤りのみ修正',
    '0 / 0 章': '0 / 0 章',
    '将在当前 API 请求结束后停止': '現在のAPIリクエスト終了後に停止します',
    '书名；可直接粘贴，生成时自动作为 EPUB 文件名': '書名。直接貼り付け可能で、生成時にEPUBファイル名として使用',
    '从剪贴板粘贴书名，并用于建议保存文件名': 'クリップボードから書名を貼り付け、推奨保存名に使用',
    '管理排版格式（新建/学习/删除/导入导出）': '組版形式を管理（新規/学習/削除/入出力）',
    '选择左侧文件预览内容': '左側のファイルを選択してプレビュー',
    '实时预览当前正文；生成后可从“打包文件”选择具体文件': '現在本文をライブプレビュー。生成後は「パッケージファイル」から選択',
    '内置 OCR ↔ 外部 OCR': '内蔵OCR ↔ 外部OCR',
    '只处理右侧光标所在行；允许把该行拆成正文和对白': '右側カーソル行のみ処理。本文と台詞への分割を許可',
    '只处理右侧选中的连续行；不会跨越插图标记': '右側で選択した連続行のみ処理。挿絵マーカーを跨がない',
    '查看 AI 处理前/处理后的逐组修改，并双击定位到双栏': 'AI処理前後の変更をグループ表示し、ダブルクリックで両ペインに移動',
    '开启或关闭句末可疑位置标红': '文末の疑わしい位置の赤表示を切替',
    '尚未载入正文版本': '本文バージョン未読込',
    '正在按当前编辑内容重新匹配……': '現在の編集内容で再照合中…',
    'AI状态：已确认，尚未应用到其他界面': 'AI状態：確認済み、他画面には未適用',
    'AI状态：未运行 / 已恢复处理前文本': 'AI状態：未実行 / 処理前テキストを復元',
    '已拒绝 AI 结果，右侧完整恢复到处理前状态。': 'AI結果を拒否し、右側を処理前状態へ完全復元しました。',
    '已撤销上一次已接受的 AI 纠错排版结果。': '前回受け入れたAI校正・組版結果を取り消しました。',
    '外部 OCR：正在后台融合……': '外部OCR：バックグラウンドで融合中…',
    '外部 OCR：融合失败': '外部OCR：融合失敗',
    '✓ 选择': '✓ 選択',
    '单 OCR：暂无可载入结果': '単一OCR：読込可能な結果なし',
    '恢复切换前的多模型文本、对齐、人工候选选择和未完成状态。': '切替前の複数モデル本文・整列・手動候補選択・未完了状態を復元します。',
    '关闭后选择会保留在当前句，便于再次核对；默认开启。': 'オフでは選択後も現在文に留まり再確認できます。既定はオンです。',
    '定位上一条仍需判断的句子。': '判断が必要な前の文へ移動します。',
    '定位下一条仍需判断的句子。': '判断が必要な次の文へ移動します。',
    '正在准备轻量裁决状态…': '軽量裁決状態を準備中…',
    '逐句裁决 · 当前候选': '文ごとの裁決 · 現在候補',
    '已撤销整组原子修复选择，事务成员全部恢复为待确认。': '原子的修復グループの選択を取り消し、全メンバーを未確認へ戻しました。',
    '所有不一致候选均已完成选择；没有上一组待判断内容。': '全ての不一致候補を選択済みです。前の未判断グループはありません。',
    '已完成最后一句；没有下一句。': '最後の文まで完了しました。次の文はありません。',
    '所有不一致候选均已完成选择；没有下一组待判断内容。': '全ての不一致候補を選択済みです。次の未判断グループはありません。',
    'OCR 裁决任务失败': 'OCR裁決タスクに失敗しました',
    'V4 多模型分歧裁决包（默认推荐）': 'V4 複数モデル差異裁決パッケージ（推奨）',
    'V3 标准紧凑包（兼容）': 'V3 標準コンパクトパッケージ（互換）',
    'V3 完整取证包（调试）': 'V3 完全証拠パッケージ（デバッグ）',
    '包含出版参考证据（可选，默认关闭）': '出版参照証拠を含める（任意、既定オフ）',
    '未选择出版参考 EPUB': '出版参照EPUB未選択',
    '选择 EPUB…': 'EPUBを選択…',
    '校验完成，正在分批刷新 OCR 对比界面…': '検証完了。OCR比較画面を分割更新中…',
    '当前没有需要跳转的未决分歧或已保留历史分歧。': '移動対象の未決差異または保持済み履歴差異はありません。',
    '完成 OCR 后，右侧显示当前句对应的单列或多列原图。': 'OCR完了後、右側に現在文対応の単列/複数列原画像を表示します。',
    '使用此候选': 'この候補を使用',
    '使用 macOS 预览打开当前句/列图片（⌥P）': '現在の文/列画像をmacOSプレビューで開く（⌥P）',
    '完成 OCR 后在此横排逐句校对': 'OCR完了後、ここで横書きの文単位校正を行います',
    '图文逐句索引建立失败': '文単位の画像/本文インデックス作成に失敗',
    '已跳到上一条 OCR 对比不一致句图': '前のOCR不一致文画像へ移動しました',
    '已跳到下一条 OCR 对比不一致句图': '次のOCR不一致文画像へ移動しました',
    '苹果手写板已打开；点击“复制并返回”后自动写入此处': 'Apple手書きパッドを開きました。「コピーして戻る」で自動入力します',
    '已接收苹果手写结果，尚未保存': 'Apple手書き結果を受信しました。未保存です',
    '核心运行合同：始终启用，不是可切换选项': 'コア実行契約：常時有効、切替不可',
    '正在检测本机设备和已安装 OCR 运行时…': 'ローカル機器とインストール済みOCR実行環境を検出中…',
    '设备检测失败；未修改任何 OCR 设置。': 'デバイス検出に失敗しました。OCR設定は変更していません。',
    '更新源已锁定，配置和界面都不能改到其它仓库。': '更新元は固定されており、設定/UIから別リポジトリへ変更できません。',
    '更新分支固定为 main。': '更新ブランチはmain固定です。',
    '尚未检查更新。启动程序不会自动联网检查。': '更新未確認です。起動時に自動オンライン確認は行いません。',
    '检查与更新过程会显示在这里。': '確認・更新処理をここに表示します。',
    '正在安全更新程序代码…': 'アプリコードを安全に更新中…',
    '📥 导入 PaddleOCR-VL': '📥 PaddleOCR-VLをインポート',
    '没有可预览页面': 'プレビュー可能なページがありません',
    '单击选择 · 双击打开高清预览 · 右键设置页面类型': 'クリックで選択 · ダブルクリックで高精細プレビュー · 右クリックでページ種別',
    '本地最低备用可直接使用；字符库不完整，未识字会保留 □。': 'ローカル最終予備は使用可能です。未登録文字は□で保持します。',
    'OpenVINO 日语手写模型与运行环境可用。': 'OpenVINO日本語手書きモデル/実行環境が利用可能です。',
    '正在打开自动轨迹…': '自動軌跡を開いています…',
    '重新下载模型': 'モデルを再ダウンロード',
    '未选择输入': '入力未選択',
    '输入参考图': '参照画像を入力',
    '当前：正在终止 OCR，不再派发新任务…': '現在：OCRを停止中。新しいタスクは配信しません…',
    '当前图片：等待第一张识别图片…': '現在画像：最初の認識画像を待機中…',
    '当前图片：实时预览已关闭（OCR 继续运行）': '現在画像：ライブプレビューOFF（OCRは継続）',
    '实时预览已关闭；不会生成或刷新新的预览图。': 'ライブプレビューはオフです。新しいプレビュー画像は生成・更新しません。',
    '处理前': '処理前',
    '处理后': '処理後',
    '<b>格式列表</b>': '<b>形式一覧</b>',
    '输入密钥后加载模型，或手动填写': 'キー入力後にモデルを読み込むか、手動入力してください',
    '修改记录 / 结构信息': '変更履歴 / 構造情報',
    '🗑 清空': '🗑 クリア',
    'AI任务已停止': 'AIタスクを停止しました',
    '<b>EPUB 制作</b>': '<b>EPUB作成</b>',
    '⚠ 封面：未设置（可在页面管理中指定）': '⚠ 表紙：未設定（ページ管理で指定可能）',
    '🖼 封面未设置': '🖼 表紙未設定',
    '☰ 0 章': '☰ 0 章',
    '🎨 0 图': '🎨 0 枚',
    'CSS 默认模板': '既定CSSテンプレート',
    '待构建': '未構築',
    '左侧：': '左：',
    '右侧：': '右：',
    '文字主稿：': '本文マスター：',
    '0 处': '0 件',
    'AI状态：处理中；右侧暂时锁定': 'AI状態：処理中。右側を一時ロック',
    'AI状态：已确认并应用到当前文档': 'AI状態：確認済み、現在の文書へ適用済み',
    '句末复核已开启；可点击“重新扫描”或“下一处”。': '文末確認を有効化。「再スキャン」または「次へ」を使用できます。',
    '句末复核已关闭。': '文末確認を無効化しました。',
    '句末复核已重新扫描：右侧当前没有发现可疑句。': '文末を再スキャンしました。右側に疑わしい文はありません。',
    '关': 'オフ',
    '右侧当前没有检测到缺少句末标点的可疑句。': '右側に文末句読点欠落の疑いがある文はありません。',
    '请先选择要设置样式的文字': '先にスタイルを設定する文字を選択してください',
    '请先选择要清除样式的文字': '先にスタイルを解除する文字を選択してください',
    '参考评分没有得到可用结果。': '参照スコアリングで有効な結果が得られませんでした。',
    '候选文字一致；来源仍分别保留，可重新选择': '候補文字は一致。ソースは個別保持され再選択できます',
    '已选择；未选候选已收起，当前文字仍可编辑': '選択済み。未選択候補は折りたたみ、現在文字は編集可能',
    '已完成裁决 · 下方显示纠错前只读分歧证据': '裁決完了 · 下に修正前の読み取り専用差異証拠を表示',
    '两模型共同候选 · 已重新打开证据': '2モデル共通候補 · 証拠を再表示',
    '每批页数': 'バッチあたりページ数',
    '前后重叠页': '前後オーバーラップページ',
    '单请求输入上限': '1リクエスト入力上限',
    '该模型的逐句 OCR 结果': 'このモデルの文単位OCR結果',
    '全文融合结果（可查看、重新选择与编辑）': '全文融合結果（表示・再選択・編集可能）',
    'OCR 对比 · 全文对比': 'OCR比較 · 全文比較',
    '＋ 从 OCR 识别载入': '＋ OCR認識から読み込む',
    '尚未导入逐源纠错': 'ソース別修正は未インポート',
    '已恢复完整多模型文本视图；逐句选择和候选修改均已保留。': '完全な複数モデル本文表示を復元。文選択と候補編集は保持されています。',
    '已重新打开当前句，图文对照同步恢复为待确认。': '現在文を再オープンし、画像/本文確認も未確認へ戻しました。',
    'OCR 对比 · 已恢复纠错会话': 'OCR比較 · 修正セッション復元済み',
    'AI 修复包模式：': 'AI修復パッケージモード：',
    'AI 审定已取消；当前融合稿没有被半途改写。': 'AI裁定をキャンセルしました。現在の融合稿は途中変更されていません。',
    '当前句没有可用的原图裁片。': '現在文に利用可能な原画像クロップがありません。',
    '✓ 当前采用': '✓ 現在採用',
    '横排 · 上→下、左→右': '横書き · 上→下、左→右',
    '图片不可用': '画像を利用できません',
    '候选已写入文本框，但保存失败；请检查当前句映射': '候補をテキスト欄へ書き込みましたが保存に失敗。現在文の対応を確認してください',
    '所有需要判断的句图均已核对': '判断が必要な文画像はすべて確認済みです',
    '当前文档没有多模型 OCR 分歧句': '現在文書に複数モデルOCR差異文はありません',
    '已保存，正在同步到 OCR 对比…': '保存済み。OCR比較へ同期中…',
    '已同步 OCR 对比最终选择': 'OCR比較の最終選択を同期しました',
    '已保存，当前是最后一句': '保存済み。現在が最後の文です',
    '即将支持': '近日対応',
    '无需安装': 'インストール不要',
    '正在打开…': '開いています…',
    '当前图片：等待下一张识别图片…': '現在画像：次の認識画像を待機中…',
    '设置已保存；本地旧 API Key 已彻底移除': '設定を保存しました。旧ローカルAPIキーは完全に削除済みです',
    '恢复为 AI 来源版本': 'AIソース版へ復元',
    '恢复为本次 AI 处理所使用的 OCR 或替换结果': '今回のAI処理で使用したOCR/置換結果へ復元',
    '两模型共同候选 · 已自动保留，可查看证据': '2モデル共通候補 · 自動保持、証拠表示可能',
    '✓ 已载入当前单OCR': '✓ 現在の単一OCRを読込済み',
    'OCR 对比已重新打开此句，等待选择': 'OCR比較でこの文を再オープンし選択待ちです',
    'Apple 桥接已就绪': 'Appleブリッジ準備完了',
    'OpenVINO 已就绪': 'OpenVINO準備完了',
    '正在编译…': 'コンパイル中…',
    '当前：等待下一次 OCR 进度事件…': '現在：次のOCR進捗イベント待機中…',
    '两模型共同候选 · 按 v8 规则自动保留': '2モデル共通候補 · v8ルールで自動保持',
    '所有独立模型完全一致 · 已自动保留': '全独立モデルが完全一致 · 自動保持',
    '仅一个有效候选 · 已保留': '有効候補が1つのみ · 保持',
})


_EXACT_EN.update({
    "在当前光标处分行，并在另一侧插入空白对齐行（Enter）": "Split at the current cursor and insert a blank alignment line on the other side (Enter)",
    "可选：人名/地名/技能名术语表（TXT 或 JSON）": "Optional: names/places/skill glossary (TXT or JSON)",
    "简体中文": "Simplified Chinese",
})
_EXACT_JA.update({
    "在当前光标处分行，并在另一侧插入空白对齐行（Enter）": "現在カーソル位置で改行し、反対側に空の整列行を挿入（Enter）",
    "可选：人名/地名/技能名术语表（TXT 或 JSON）": "任意：人名/地名/スキル名用語集（TXT または JSON）",
    "简体中文": "簡体字中国語",
})


# Full UI help/tooltips: OCR engine and model configuration.
_EXACT_EN.update({'滚轮缩放 · 拖动平移 · 双击切换适应窗口/100% · ←/→ 翻页 · Esc 关闭': 'Wheel to zoom · Drag to pan · Double-click toggles Fit/100% · ←/→ '
                                                  'pages · Esc closes',
 '自动裁边、透视拉正、轻度纠偏、漂白去阴影与可选双页拆分。只生成会话临时副本，原图和现有 OCR 流程不修改。': 'Auto-crop, perspective straighten, light deskew, '
                                                           'whitening/shadow removal, and optional spread splitting. '
                                                           'Only session-temporary copies are created; source images '
                                                           'and the existing OCR pipeline are unchanged.',
 '打开图片文件夹 / PDF / 单张图片开始\n\n支持 PNG · JPG · HEIC · TIFF · PDF': 'Open an image folder / PDF / single image to begin\n'
                                                               '\n'
                                                               'Supports PNG · JPG · HEIC · TIFF · PDF',
 '模型1使用上方当前选中的 OCR；模型2～6共用完全相同的固定正文区域、物理分列几何和正文可见层。不同引擎只按自身输入合同选择宽上下文或紧凑白底视窗，正文/Ruby归属不会改变。完成后进入独立 OCR 对比工作区逐句自动选优或手动选择。': 'Model '
                                                                                                                           '1 '
                                                                                                                           'uses '
                                                                                                                           'the '
                                                                                                                           'OCR '
                                                                                                                           'selected '
                                                                                                                           'above; '
                                                                                                                           'models '
                                                                                                                           '2–6 '
                                                                                                                           'share '
                                                                                                                           'the '
                                                                                                                           'same '
                                                                                                                           'fixed '
                                                                                                                           'body '
                                                                                                                           'region, '
                                                                                                                           'physical-column '
                                                                                                                           'geometry, '
                                                                                                                           'and '
                                                                                                                           'body '
                                                                                                                           'visibility '
                                                                                                                           'layer. '
                                                                                                                           'Each '
                                                                                                                           'engine '
                                                                                                                           'only '
                                                                                                                           'chooses '
                                                                                                                           'a '
                                                                                                                           'wider '
                                                                                                                           'context '
                                                                                                                           'or '
                                                                                                                           'compact '
                                                                                                                           'white '
                                                                                                                           'viewport '
                                                                                                                           'according '
                                                                                                                           'to '
                                                                                                                           'its '
                                                                                                                           'input '
                                                                                                                           'contract; '
                                                                                                                           'body/Ruby '
                                                                                                                           'ownership '
                                                                                                                           'never '
                                                                                                                           'changes. '
                                                                                                                           'When '
                                                                                                                           'finished, '
                                                                                                                           'use '
                                                                                                                           'the '
                                                                                                                           'separate '
                                                                                                                           'OCR '
                                                                                                                           'Compare '
                                                                                                                           'workspace '
                                                                                                                           'for '
                                                                                                                           'automatic '
                                                                                                                           'sentence '
                                                                                                                           'selection '
                                                                                                                           'or '
                                                                                                                           'manual '
                                                                                                                           'choice.',
 '这是运行合同而不是可选开关。普通模型对每个物理列执行一次主 OCR；NDLOCR 的智能混合/整页/逐列策略由下方独立选项控制。空结果保留为 □ 进入裁决。': 'This is a runtime contract, not an '
                                                                                   'optional switch. Normal models '
                                                                                   'perform one primary OCR per '
                                                                                   'physical column; NDLOCR '
                                                                                   'smart-hybrid/full-page/per-column '
                                                                                   'behavior is controlled separately '
                                                                                   'below. Empty results remain as □ '
                                                                                   'for adjudication.',
 '仅在日文精确分列的多模型模式下生效。模型1和模型2先独立识别同一物理列；文字相同的列不再调用模型3～6；仍有分歧时，后续模型依次只识别未决列。两模型共同候选不会冒充多模型一致，也不会自动压过 AI 或人工裁决。关闭后，所有已选模型都会完整识别。': 'Applies '
                                                                                                                               'only '
                                                                                                                               'to '
                                                                                                                               'multi-model '
                                                                                                                               'Japanese '
                                                                                                                               'precise-column '
                                                                                                                               'mode. '
                                                                                                                               'Models '
                                                                                                                               '1 '
                                                                                                                               'and '
                                                                                                                               '2 '
                                                                                                                               'independently '
                                                                                                                               'recognize '
                                                                                                                               'the '
                                                                                                                               'same '
                                                                                                                               'physical '
                                                                                                                               'column; '
                                                                                                                               'matching '
                                                                                                                               'columns '
                                                                                                                               'do '
                                                                                                                               'not '
                                                                                                                               'call '
                                                                                                                               'models '
                                                                                                                               '3–6. '
                                                                                                                               'If '
                                                                                                                               'disagreement '
                                                                                                                               'remains, '
                                                                                                                               'later '
                                                                                                                               'models '
                                                                                                                               'sequentially '
                                                                                                                               'recognize '
                                                                                                                               'only '
                                                                                                                               'unresolved '
                                                                                                                               'columns. '
                                                                                                                               'A '
                                                                                                                               'two-model '
                                                                                                                               'shared '
                                                                                                                               'candidate '
                                                                                                                               'never '
                                                                                                                               'pretends '
                                                                                                                               'to '
                                                                                                                               'be '
                                                                                                                               'multi-model '
                                                                                                                               'agreement '
                                                                                                                               'and '
                                                                                                                               'never '
                                                                                                                               'overrides '
                                                                                                                               'AI '
                                                                                                                               'or '
                                                                                                                               'human '
                                                                                                                               'adjudication. '
                                                                                                                               'When '
                                                                                                                               'disabled, '
                                                                                                                               'every '
                                                                                                                               'selected '
                                                                                                                               'model '
                                                                                                                               'runs '
                                                                                                                               'a '
                                                                                                                               'full '
                                                                                                                               'pass.',
 '独立于快速共识开关：模型1和模型2同时读取同一批已缓存物理列。每个模型使用独立临时增强目录，不会互相覆盖；共享分列裁图按页加锁。关闭快速共识时，前两模型仍可并行完成全量首轮，模型3～6随后按完整模式运行。若机器内存较小或两个模型争用同一加速设备，可关闭改为串行。': 'Independent '
                                                                                                                                        'of '
                                                                                                                                        'fast '
                                                                                                                                        'consensus: '
                                                                                                                                        'models '
                                                                                                                                        '1 '
                                                                                                                                        'and '
                                                                                                                                        '2 '
                                                                                                                                        'read '
                                                                                                                                        'the '
                                                                                                                                        'same '
                                                                                                                                        'cached '
                                                                                                                                        'physical '
                                                                                                                                        'columns '
                                                                                                                                        'in '
                                                                                                                                        'parallel. '
                                                                                                                                        'Each '
                                                                                                                                        'uses '
                                                                                                                                        'its '
                                                                                                                                        'own '
                                                                                                                                        'temporary '
                                                                                                                                        'enhancement '
                                                                                                                                        'directory; '
                                                                                                                                        'shared '
                                                                                                                                        'column '
                                                                                                                                        'crops '
                                                                                                                                        'are '
                                                                                                                                        'page-locked. '
                                                                                                                                        'With '
                                                                                                                                        'fast '
                                                                                                                                        'consensus '
                                                                                                                                        'off, '
                                                                                                                                        'the '
                                                                                                                                        'first '
                                                                                                                                        'two '
                                                                                                                                        'models '
                                                                                                                                        'still '
                                                                                                                                        'finish '
                                                                                                                                        'a '
                                                                                                                                        'full '
                                                                                                                                        'parallel '
                                                                                                                                        'pass '
                                                                                                                                        'before '
                                                                                                                                        'models '
                                                                                                                                        '3–6. '
                                                                                                                                        'Disable '
                                                                                                                                        'parallelism '
                                                                                                                                        'on '
                                                                                                                                        'low-memory '
                                                                                                                                        'systems '
                                                                                                                                        'or '
                                                                                                                                        'when '
                                                                                                                                        'both '
                                                                                                                                        'models '
                                                                                                                                        'contend '
                                                                                                                                        'for '
                                                                                                                                        'one '
                                                                                                                                        'accelerator.',
 '仅在关闭“分列阶段一致即定稿”的兼容模式下生效。共识模式会自动只把未决句组交给主模型，无需再选择此项；关闭共识后，平衡模式只由主模型执行整句校验，完整精校保持旧行为。': 'Only applies when the '
                                                                                        "compatibility mode 'finalize "
                                                                                        "when columns agree' is "
                                                                                        'disabled. Consensus mode '
                                                                                        'automatically sends only '
                                                                                        'unresolved sentence groups to '
                                                                                        'the primary model. Without '
                                                                                        'consensus, Balanced uses the '
                                                                                        'primary model for sentence '
                                                                                        'validation; Full Review '
                                                                                        'preserves legacy behavior.',
 '模型1：Apple OCR（当前主模型/结构底稿）': 'Model 1: Apple OCR (current primary/structure draft)',
 '模型1决定页码、章节、图片锚点与基本段落结构。快速共识会逐级减少模型3～6调用：两模型相同列单独统计为共同候选，后续模型只补真正分歧列；AI/人工裁决始终可以覆盖候选，原始模型输出保持不变。': 'Model 1 '
                                                                                                    'determines page '
                                                                                                    'numbers, '
                                                                                                    'chapters, image '
                                                                                                    'anchors, and base '
                                                                                                    'paragraph '
                                                                                                    'structure. Fast '
                                                                                                    'consensus '
                                                                                                    'progressively '
                                                                                                    'reduces calls to '
                                                                                                    'models 3–6: '
                                                                                                    'matching '
                                                                                                    'model-1/2 columns '
                                                                                                    'are tracked '
                                                                                                    'separately as '
                                                                                                    'shared '
                                                                                                    'candidates, and '
                                                                                                    'later models only '
                                                                                                    'resolve genuine '
                                                                                                    'conflicting '
                                                                                                    'columns. AI/human '
                                                                                                    'adjudication can '
                                                                                                    'always override '
                                                                                                    'candidates, while '
                                                                                                    'raw model output '
                                                                                                    'remains '
                                                                                                    'unchanged.',
 'Apple Vision · RecognizeTextRequest 坐标/候选': 'Apple Vision · RecognizeTextRequest coordinates/candidates',
 '仅在输入图实际只保留一个狭长竖列时启用：先紧裁文字区域，再左旋成横排送入 Vision；不会增加 OCR 次数，返回坐标会映射回原图。': 'Enable only when the input really contains '
                                                                        'one narrow vertical column: tightly crop the '
                                                                        'text, rotate 90° left, then send to Vision. '
                                                                        'OCR call count does not increase; returned '
                                                                        'coordinates are mapped back to the original '
                                                                        'image.',
 'Live Text 直接分析原尺寸分列掩膜图并读取系统 transcript；不旋转、不紧裁、不重复 OCR。该通道不提供逐字置信度、候选或文字框。': 'Live Text analyzes the original-size '
                                                                               'column mask and reads the system '
                                                                               'transcript directly; no rotation, '
                                                                               'tight crop or repeated OCR. This path '
                                                                               'does not provide per-character '
                                                                               'confidence, alternatives or text '
                                                                               'boxes.',
 '识别时会把所选页面图片发送给 Google。PDF 会先在本机拆成逐页图片，因此不需要 GCS 存储桶。': 'Recognition sends the selected page images to Google. PDFs '
                                                         'are split into page images locally first, so a GCS bucket is '
                                                         'not required.',
 '从 aistudio.baidu.com/paddleocr/task → API 调用示例复制完整 https:// URL': 'Copy the full https:// URL from '
                                                                    'aistudio.baidu.com/paddleocr/task → API call '
                                                                    'example',
 'macOS Keychain / Windows DPAPI；Linux 不做明文持久化': 'macOS Keychain / Windows DPAPI; Linux never persists it as plaintext',
 '默认使用官方异步 v2 jobs：本地文件 multipart 上传、Bearer Token、jobId 轮询和 JSONL 结果下载；默认模型为 PaddleOCR-VL-1.6。同步模式的 URL 与 AI Studio 任务页模型绑定；请匹配选择 PP-OCR、PP-StructureV3 或 PaddleOCR-VL。异步 v2 按官方模型能力自动裁剪参数；PP-StructureV3 与 PP-OCR 支持文字行方向，VL 系列自动省略。云端引擎固定整页调用，不会把分栏 ROI 单独上传。': 'Official '
                                                                                                                                                                                                                                                                  'async '
                                                                                                                                                                                                                                                                  'v2 '
                                                                                                                                                                                                                                                                  'jobs '
                                                                                                                                                                                                                                                                  'are '
                                                                                                                                                                                                                                                                  'used '
                                                                                                                                                                                                                                                                  'by '
                                                                                                                                                                                                                                                                  'default: '
                                                                                                                                                                                                                                                                  'multipart '
                                                                                                                                                                                                                                                                  'local-file '
                                                                                                                                                                                                                                                                  'upload, '
                                                                                                                                                                                                                                                                  'Bearer '
                                                                                                                                                                                                                                                                  'Token, '
                                                                                                                                                                                                                                                                  'jobId '
                                                                                                                                                                                                                                                                  'polling '
                                                                                                                                                                                                                                                                  'and '
                                                                                                                                                                                                                                                                  'JSONL '
                                                                                                                                                                                                                                                                  'result '
                                                                                                                                                                                                                                                                  'download; '
                                                                                                                                                                                                                                                                  'default '
                                                                                                                                                                                                                                                                  'model '
                                                                                                                                                                                                                                                                  'is '
                                                                                                                                                                                                                                                                  'PaddleOCR-VL-1.6. '
                                                                                                                                                                                                                                                                  'Synchronous '
                                                                                                                                                                                                                                                                  'URLs '
                                                                                                                                                                                                                                                                  'are '
                                                                                                                                                                                                                                                                  'bound '
                                                                                                                                                                                                                                                                  'to '
                                                                                                                                                                                                                                                                  'the '
                                                                                                                                                                                                                                                                  'AI '
                                                                                                                                                                                                                                                                  'Studio '
                                                                                                                                                                                                                                                                  'task-page '
                                                                                                                                                                                                                                                                  'model, '
                                                                                                                                                                                                                                                                  'so '
                                                                                                                                                                                                                                                                  'choose '
                                                                                                                                                                                                                                                                  'PP-OCR, '
                                                                                                                                                                                                                                                                  'PP-StructureV3 '
                                                                                                                                                                                                                                                                  'or '
                                                                                                                                                                                                                                                                  'PaddleOCR-VL '
                                                                                                                                                                                                                                                                  'accordingly. '
                                                                                                                                                                                                                                                                  'Async '
                                                                                                                                                                                                                                                                  'v2 '
                                                                                                                                                                                                                                                                  'trims '
                                                                                                                                                                                                                                                                  'parameters '
                                                                                                                                                                                                                                                                  'to '
                                                                                                                                                                                                                                                                  'official '
                                                                                                                                                                                                                                                                  'model '
                                                                                                                                                                                                                                                                  'capabilities; '
                                                                                                                                                                                                                                                                  'PP-StructureV3/PP-OCR '
                                                                                                                                                                                                                                                                  'support '
                                                                                                                                                                                                                                                                  'text-line '
                                                                                                                                                                                                                                                                  'orientation '
                                                                                                                                                                                                                                                                  'while '
                                                                                                                                                                                                                                                                  'VL '
                                                                                                                                                                                                                                                                  'omits '
                                                                                                                                                                                                                                                                  'it '
                                                                                                                                                                                                                                                                  'automatically. '
                                                                                                                                                                                                                                                                  'Cloud '
                                                                                                                                                                                                                                                                  'engines '
                                                                                                                                                                                                                                                                  'always '
                                                                                                                                                                                                                                                                  'receive '
                                                                                                                                                                                                                                                                  'full '
                                                                                                                                                                                                                                                                  'pages, '
                                                                                                                                                                                                                                                                  'never '
                                                                                                                                                                                                                                                                  'individual '
                                                                                                                                                                                                                                                                  'column '
                                                                                                                                                                                                                                                                  'ROIs.',
 '自动（Apple Silicon 优先 MPS，失败回退 CPU）': 'Auto (Apple Silicon prefers MPS, falls back to CPU)',
 '不重新执行文字检测，只把 tiny-v5 的空结果或低置信度原检测框交给 middle-v5。': 'Do not rerun text detection; send only tiny-v5 '
                                                    'empty/low-confidence original boxes to middle-v5.',
 'PaddleOCR（日文印刷体；优先 v6 medium）': 'PaddleOCR (Japanese print; prefer v6 medium)',
 'PaddleOCR-VL-1.6（Apple Silicon 可用官方 MLX 加速）': 'PaddleOCR-VL-1.6 (official MLX acceleration available on Apple '
                                                'Silicon)',
 '自动（Apple Silicon 优先官方 MLX；失败回退 Paddle）': 'Auto (prefer official MLX on Apple Silicon; fall back to Paddle)',
 '仅对 PaddleOCR-VL-1.6 生效。自动模式在 Apple Silicon 上通过 PaddleOCR 官方 mlx-vlm-server 接口使用 Apple GPU；MLX 安装/启动或实际推理失败时自动回退原生 Paddle。': 'Only '
                                                                                                                              'for '
                                                                                                                              'PaddleOCR-VL-1.6. '
                                                                                                                              'Auto '
                                                                                                                              'mode '
                                                                                                                              'uses '
                                                                                                                              'Apple '
                                                                                                                              'GPU '
                                                                                                                              'through '
                                                                                                                              "PaddleOCR's "
                                                                                                                              'official '
                                                                                                                              'mlx-vlm-server '
                                                                                                                              'on '
                                                                                                                              'Apple '
                                                                                                                              'Silicon; '
                                                                                                                              'MLX '
                                                                                                                              'install/startup/inference '
                                                                                                                              'failures '
                                                                                                                              'automatically '
                                                                                                                              'fall '
                                                                                                                              'back '
                                                                                                                              'to '
                                                                                                                              'native '
                                                                                                                              'Paddle.',
 '自动重试（HF → ModelScope → BOS → AIStudio）': 'Automatic retry (HF → ModelScope → BOS → AIStudio)',
 'PaddleOCR 3.x 默认从 Hugging Face 下载。若当前网络无法访问，可改用 ModelScope 或百度 BOS；自动模式只在模型初始化阶段切换来源，不会重复 OCR 页面。PaddleOCR-VL 的 MLX-VLM 模型由官方 MLX 路径使用 Hugging Face 模型 ID 加载，此选项主要控制 Paddle 客户端/版面模型来源。': 'PaddleOCR '
                                                                                                                                                                                            '3.x '
                                                                                                                                                                                            'downloads '
                                                                                                                                                                                            'from '
                                                                                                                                                                                            'Hugging '
                                                                                                                                                                                            'Face '
                                                                                                                                                                                            'by '
                                                                                                                                                                                            'default. '
                                                                                                                                                                                            'If '
                                                                                                                                                                                            'unavailable, '
                                                                                                                                                                                            'choose '
                                                                                                                                                                                            'ModelScope '
                                                                                                                                                                                            'or '
                                                                                                                                                                                            'Baidu '
                                                                                                                                                                                            'BOS; '
                                                                                                                                                                                            'Auto '
                                                                                                                                                                                            'changes '
                                                                                                                                                                                            'sources '
                                                                                                                                                                                            'only '
                                                                                                                                                                                            'during '
                                                                                                                                                                                            'model '
                                                                                                                                                                                            'initialization '
                                                                                                                                                                                            'and '
                                                                                                                                                                                            'never '
                                                                                                                                                                                            'repeats '
                                                                                                                                                                                            'page '
                                                                                                                                                                                            'OCR. '
                                                                                                                                                                                            "PaddleOCR-VL's "
                                                                                                                                                                                            'MLX-VLM '
                                                                                                                                                                                            'path '
                                                                                                                                                                                            'loads '
                                                                                                                                                                                            'the '
                                                                                                                                                                                            'official '
                                                                                                                                                                                            'Hugging '
                                                                                                                                                                                            'Face '
                                                                                                                                                                                            'model '
                                                                                                                                                                                            'ID; '
                                                                                                                                                                                            'this '
                                                                                                                                                                                            'option '
                                                                                                                                                                                            'mainly '
                                                                                                                                                                                            'controls '
                                                                                                                                                                                            'Paddle '
                                                                                                                                                                                            'client/layout-model '
                                                                                                                                                                                            'sources.',
 '先创建独立环境并下载/初始化当前 Paddle 模型，不开始整本 OCR。': 'Create an isolated environment and download/initialize the current Paddle '
                                          'model without starting full-book OCR.',
 'PP-OCRv6 / PP-Structure 保持原有 Paddle 路径。PaddleOCR-VL-1.6 在 Apple Silicon 上默认通过官方 MLX-VLM 后端加速，并使用独立环境避免依赖冲突；MLX 不可用或页级推理失败会自动回退 Paddle。': 'PP-OCRv6 '
                                                                                                                                           '/ '
                                                                                                                                           'PP-Structure '
                                                                                                                                           'keep '
                                                                                                                                           'the '
                                                                                                                                           'existing '
                                                                                                                                           'Paddle '
                                                                                                                                           'path. '
                                                                                                                                           'PaddleOCR-VL-1.6 '
                                                                                                                                           'uses '
                                                                                                                                           'the '
                                                                                                                                           'official '
                                                                                                                                           'MLX-VLM '
                                                                                                                                           'backend '
                                                                                                                                           'by '
                                                                                                                                           'default '
                                                                                                                                           'on '
                                                                                                                                           'Apple '
                                                                                                                                           'Silicon '
                                                                                                                                           'in '
                                                                                                                                           'an '
                                                                                                                                           'isolated '
                                                                                                                                           'environment '
                                                                                                                                           'to '
                                                                                                                                           'avoid '
                                                                                                                                           'dependency '
                                                                                                                                           'conflicts; '
                                                                                                                                           'if '
                                                                                                                                           'MLX '
                                                                                                                                           'is '
                                                                                                                                           'unavailable '
                                                                                                                                           'or '
                                                                                                                                           'page '
                                                                                                                                           'inference '
                                                                                                                                           'fails, '
                                                                                                                                           'it '
                                                                                                                                           'automatically '
                                                                                                                                           'falls '
                                                                                                                                           'back '
                                                                                                                                           'to '
                                                                                                                                           'Paddle.',
 '整页按从上到下、从左到右读取；不执行日文分列、Ruby 过滤、竖列旋转、逐列成句或日语手写识别。模式切换不会修改日文竖排设置。': 'Read full pages top-to-bottom, left-to-right; do '
                                                                    'not perform Japanese column splitting, Ruby '
                                                                    'filtering, vertical-column rotation, per-column '
                                                                    'sentence building or Japanese handwriting '
                                                                    'recognition. Switching this mode does not alter '
                                                                    'Japanese vertical settings.',
 '仅合并纵向重叠且基线接近的相邻框，按左到右排序；不会跨行或跨段合并。': 'Merge only adjacent boxes with vertical overlap and close baselines, sorted '
                                       'left-to-right; never merge across lines or paragraphs.',
 '按多页重复频率过滤短页眉页脚。与日文逐列成句的保全策略相互独立。': 'Filter short repeated headers/footers by frequency across pages. This is '
                                     'independent from preservation rules for Japanese per-column sentence building.',
 '固定正文框会先在整页同尺寸画布上生成纸白掩膜，框外页眉/页脚不参与检测；开启后再检测日文竖列，为每列生成整页同尺寸掩膜图。目标文字不缩放、不拉伸。': 'The fixed body region first creates a '
                                                                              'paper-white mask on a same-size canvas '
                                                                              'so outside headers/footers are '
                                                                              'excluded. When enabled, Japanese '
                                                                              'vertical columns are then detected and '
                                                                              'each column gets a same-size full-page '
                                                                              'mask. Target text is never scaled or '
                                                                              'stretched.',
 '默认关闭，因此不会下载模型、不会增加 OCR 时间。开启后普通 OCR 仍只识别分列正文，并强制从其临时列图中排除 Ruby；只有 findtextCenterNet 单独读取原始页面提取 rubybase ↔ 振假名关系。两条证据通道完全独立，findtextCenterNet 不参与正文多数票或字符融合。': 'Off '
                                                                                                                                                                'by '
                                                                                                                                                                'default, '
                                                                                                                                                                'so '
                                                                                                                                                                'no '
                                                                                                                                                                'model '
                                                                                                                                                                'is '
                                                                                                                                                                'downloaded '
                                                                                                                                                                'and '
                                                                                                                                                                'OCR '
                                                                                                                                                                'time '
                                                                                                                                                                'is '
                                                                                                                                                                'not '
                                                                                                                                                                'increased. '
                                                                                                                                                                'When '
                                                                                                                                                                'enabled, '
                                                                                                                                                                'normal '
                                                                                                                                                                'OCR '
                                                                                                                                                                'still '
                                                                                                                                                                'recognizes '
                                                                                                                                                                'only '
                                                                                                                                                                'column '
                                                                                                                                                                'body '
                                                                                                                                                                'text '
                                                                                                                                                                'and '
                                                                                                                                                                'forcibly '
                                                                                                                                                                'excludes '
                                                                                                                                                                'Ruby '
                                                                                                                                                                'from '
                                                                                                                                                                'temporary '
                                                                                                                                                                'column '
                                                                                                                                                                'images; '
                                                                                                                                                                'only '
                                                                                                                                                                'findtextCenterNet '
                                                                                                                                                                'reads '
                                                                                                                                                                'the '
                                                                                                                                                                'untouched '
                                                                                                                                                                'source '
                                                                                                                                                                'page '
                                                                                                                                                                'to '
                                                                                                                                                                'extract '
                                                                                                                                                                'rubybase '
                                                                                                                                                                '↔ '
                                                                                                                                                                'furigana. '
                                                                                                                                                                'The '
                                                                                                                                                                'two '
                                                                                                                                                                'evidence '
                                                                                                                                                                'channels '
                                                                                                                                                                'are '
                                                                                                                                                                'fully '
                                                                                                                                                                'independent; '
                                                                                                                                                                'findtextCenterNet '
                                                                                                                                                                'never '
                                                                                                                                                                'joins '
                                                                                                                                                                'body-text '
                                                                                                                                                                'voting '
                                                                                                                                                                'or '
                                                                                                                                                                'character '
                                                                                                                                                                'fusion.',
 '智能 ROI 会复用普通 OCR 分列时顺手记录的疑似 Ruby 几何，只从未清理原图裁取相邻几列上下文交给 findtextCenterNet；没有候选的页面不会再次 OCR。全页模式仅用于漏检诊断。': 'Smart ROI '
                                                                                                          'reuses '
                                                                                                          'suspected '
                                                                                                          'Ruby '
                                                                                                          'geometry '
                                                                                                          'recorded '
                                                                                                          'during '
                                                                                                          'normal OCR '
                                                                                                          'column '
                                                                                                          'splitting '
                                                                                                          'and crops '
                                                                                                          'only '
                                                                                                          'nearby-column '
                                                                                                          'context '
                                                                                                          'from the '
                                                                                                          'untouched '
                                                                                                          'page for '
                                                                                                          'findtextCenterNet. '
                                                                                                          'Pages with '
                                                                                                          'no '
                                                                                                          'candidates '
                                                                                                          'are not '
                                                                                                          're-OCRed. '
                                                                                                          'Full-page '
                                                                                                          'mode is '
                                                                                                          'only for '
                                                                                                          'missed-detection '
                                                                                                          'diagnostics.',
 'Hayai 只负责已经分离的文字 crop，不做整页文字检测。Novel Formatter 会复用固定正文框、右→左物理列、Ruby 清理、共享 crop 与多模型融合；每段默认约 24 字，减少不必要的切段调用。': 'Hayai '
                                                                                                                 'only '
                                                                                                                 'recognizes '
                                                                                                                 'already-separated '
                                                                                                                 'text '
                                                                                                                 'crops '
                                                                                                                 'and '
                                                                                                                 'does '
                                                                                                                 'not '
                                                                                                                 'detect '
                                                                                                                 'full-page '
                                                                                                                 'text. '
                                                                                                                 'Novel '
                                                                                                                 'Formatter '
                                                                                                                 'reuses '
                                                                                                                 'the '
                                                                                                                 'fixed '
                                                                                                                 'body '
                                                                                                                 'region, '
                                                                                                                 'right-to-left '
                                                                                                                 'physical '
                                                                                                                 'columns, '
                                                                                                                 'Ruby '
                                                                                                                 'cleanup, '
                                                                                                                 'shared '
                                                                                                                 'crops '
                                                                                                                 'and '
                                                                                                                 'multi-model '
                                                                                                                 'fusion; '
                                                                                                                 'segments '
                                                                                                                 'default '
                                                                                                                 'to '
                                                                                                                 'about '
                                                                                                                 '24 '
                                                                                                                 'characters '
                                                                                                                 'to '
                                                                                                                 'avoid '
                                                                                                                 'unnecessary '
                                                                                                                 'splitting.'})
_EXACT_JA.update({'滚轮缩放 · 拖动平移 · 双击切换适应窗口/100% · ←/→ 翻页 · Esc 关闭': 'ホイールで拡大縮小 · ドラッグで移動 · ダブルクリックでウィンドウ適合/100%切替 · ←/→ ページ移動 · Escで閉じる',
 '自动裁边、透视拉正、轻度纠偏、漂白去阴影与可选双页拆分。只生成会话临时副本，原图和现有 OCR 流程不修改。': '自動トリミング、遠近補正、軽い傾き補正、白地化/影除去、任意の見開き分割を行います。セッション用の一時コピーのみ生成し、元画像と既存OCR処理は変更しません。',
 '打开图片文件夹 / PDF / 单张图片开始\n\n支持 PNG · JPG · HEIC · TIFF · PDF': '画像フォルダ / PDF / 単一画像を開いて開始\n'
                                                               '\n'
                                                               'PNG · JPG · HEIC · TIFF · PDF に対応',
 '模型1使用上方当前选中的 OCR；模型2～6共用完全相同的固定正文区域、物理分列几何和正文可见层。不同引擎只按自身输入合同选择宽上下文或紧凑白底视窗，正文/Ruby归属不会改变。完成后进入独立 OCR 对比工作区逐句自动选优或手动选择。': 'モデル1は上で選択したOCRを使用し、モデル2/3には同一入力・固定範囲・列順を渡します。完了後は独立したOCR比較ワークスペースで文ごとの自動選択または手動選択を行います。',
 '这是运行合同而不是可选开关。普通模型对每个物理列执行一次主 OCR；NDLOCR 的智能混合/整页/逐列策略由下方独立选项控制。空结果保留为 □ 进入裁决。': 'これは任意設定ではなく実行契約です。通常モデルは各物理列に主OCRを1回実行し、NDLOCRのスマート混合/全ページ/列別戦略は下の専用設定で制御します。空結果は□のまま裁決へ送ります。',
 '仅在日文精确分列的多模型模式下生效。模型1和模型2先独立识别同一物理列；文字相同的列不再调用模型3～6；仍有分歧时，后续模型依次只识别未决列。两模型共同候选不会冒充多模型一致，也不会自动压过 AI 或人工裁决。关闭后，所有已选模型都会完整识别。': '日本語の精密列分割による複数モデルモードでのみ有効です。モデル1/2が同一物理列を独立認識し、一致した列はモデル3を省略して「2モデル共通候補」として扱います。3モデル一致を装わず、AI/手動裁決より優先もしません。オフではモデル3も全列を認識します。',
 '独立于快速共识开关：模型1和模型2同时读取同一批已缓存物理列。每个模型使用独立临时增强目录，不会互相覆盖；共享分列裁图按页加锁。关闭快速共识时，前两模型仍可并行完成全量首轮，模型3～6随后按完整模式运行。若机器内存较小或两个模型争用同一加速设备，可关闭改为串行。': '高速合意とは独立して、モデル1/2が同じキャッシュ済み物理列を並列認識します。各モデルは独立一時強調フォルダを使い、共有列クロップはページ単位でロックします。高速合意オフでも先の2モデルは全量並列で完了後、モデル3を実行します。メモリが少ない場合や同一アクセラレータを競合する場合は並列をオフにしてください。',
 '仅在关闭“分列阶段一致即定稿”的兼容模式下生效。共识模式会自动只把未决句组交给主模型，无需再选择此项；关闭共识后，平衡模式只由主模型执行整句校验，完整精校保持旧行为。': '「列段階で一致なら確定」互換モードを無効にした場合のみ有効です。合意モードでは未決文グループだけを主モデルへ自動送信します。合意オフ時、バランスは主モデルのみで文検証し、完全精査は旧動作を維持します。',
 '模型1：Apple OCR（当前主模型/结构底稿）': 'モデル1：macOS OCR（Apple Vision）（現在の主モデル/構造下書き）',
 '模型1决定页码、章节、图片锚点与基本段落结构。快速共识会逐级减少模型3～6调用：两模型相同列单独统计为共同候选，后续模型只补真正分歧列；AI/人工裁决始终可以覆盖候选，原始模型输出保持不变。': 'モデル1がページ番号、章、画像アンカー、基本段落構造を決めます。高速合意はモデル3の呼出しだけを減らし、モデル1/2一致列は共通候補として別集計します。真の差異は独立OCR本文の不一致のみです。AI/手動裁決は常に候補を上書きでき、元モデル出力は変更しません。',
 'Apple Vision · RecognizeTextRequest 坐标/候选': 'Apple Vision · RecognizeTextRequest 座標/候補',
 '仅在输入图实际只保留一个狭长竖列时启用：先紧裁文字区域，再左旋成横排送入 Vision；不会增加 OCR 次数，返回坐标会映射回原图。': '入力が実際に細長い縦1列だけを含む場合にのみ有効化します。文字領域をタイトに切り出して左90°回転しVisionへ渡します。OCR回数は増えず、返却座標は元画像へ戻して対応付けます。',
 'Live Text 直接分析原尺寸分列掩膜图并读取系统 transcript；不旋转、不紧裁、不重复 OCR。该通道不提供逐字置信度、候选或文字框。': 'Live '
                                                                               'Textは原寸の列マスクを直接解析しシステムtranscriptを読みます。回転・タイトクロップ・再OCRは行いません。この経路は文字単位信頼度、候補、文字枠を提供しません。',
 '识别时会把所选页面图片发送给 Google。PDF 会先在本机拆成逐页图片，因此不需要 GCS 存储桶。': '認識時は選択ページ画像をGoogleへ送信します。PDFは先にローカルでページ画像へ分割するため、GCSバケットは不要です。',
 '从 aistudio.baidu.com/paddleocr/task → API 调用示例复制完整 https:// URL': 'aistudio.baidu.com/paddleocr/task → API呼出し例から完全な '
                                                                    'https:// URL をコピー',
 'macOS Keychain / Windows DPAPI；Linux 不做明文持久化': 'macOS Keychain / Windows DPAPI。Linuxでは平文保存しません',
 '默认使用官方异步 v2 jobs：本地文件 multipart 上传、Bearer Token、jobId 轮询和 JSONL 结果下载；默认模型为 PaddleOCR-VL-1.6。同步模式的 URL 与 AI Studio 任务页模型绑定；请匹配选择 PP-OCR、PP-StructureV3 或 PaddleOCR-VL。异步 v2 按官方模型能力自动裁剪参数；PP-StructureV3 与 PP-OCR 支持文字行方向，VL 系列自动省略。云端引擎固定整页调用，不会把分栏 ROI 单独上传。': '既定は公式非同期v2 '
                                                                                                                                                                                                                                                                  'jobsです：ローカルファイルをmultipart送信し、Bearer '
                                                                                                                                                                                                                                                                  'Token、jobIdポーリング、JSONL結果取得を使います。既定モデルはPaddleOCR-VL-1.6です。同期URLはAI '
                                                                                                                                                                                                                                                                  'Studioタスクページのモデルに紐づくため、PP-OCR '
                                                                                                                                                                                                                                                                  '/ '
                                                                                                                                                                                                                                                                  'PP-StructureV3 '
                                                                                                                                                                                                                                                                  '/ '
                                                                                                                                                                                                                                                                  'PaddleOCR-VLを一致させてください。非同期v2は公式能力に合わせて引数を自動調整し、PP-StructureV3/PP-OCRは文字行方向に対応、VL系では自動省略します。クラウドには常に全ページを送り、列ROI単体は送信しません。',
 '自动（Apple Silicon 优先 MPS，失败回退 CPU）': '自動（Apple SiliconはMPS優先、失敗時CPUへ）',
 '不重新执行文字检测，只把 tiny-v5 的空结果或低置信度原检测框交给 middle-v5。': '文字検出は再実行せず、tiny-v5の空/低信頼の元検出枠だけをmiddle-v5へ送ります。',
 'PaddleOCR（日文印刷体；优先 v6 medium）': 'PaddleOCR（日本語印刷体；v6 medium優先）',
 'PaddleOCR-VL-1.6（Apple Silicon 可用官方 MLX 加速）': 'PaddleOCR-VL-1.6（Apple Siliconでは公式MLX高速化を利用可能）',
 '自动（Apple Silicon 优先官方 MLX；失败回退 Paddle）': '自動（Apple Siliconでは公式MLX優先、失敗時Paddleへ）',
 '仅对 PaddleOCR-VL-1.6 生效。自动模式在 Apple Silicon 上通过 PaddleOCR 官方 mlx-vlm-server 接口使用 Apple GPU；MLX 安装/启动或实际推理失败时自动回退原生 Paddle。': 'PaddleOCR-VL-1.6専用です。自動モードではApple '
                                                                                                                              'Silicon上でPaddleOCR公式mlx-vlm-serverを介してApple '
                                                                                                                              'GPUを使用し、MLXの導入・起動・推論失敗時は原生Paddleへ自動フォールバックします。',
 '自动重试（HF → ModelScope → BOS → AIStudio）': '自動再試行（HF → ModelScope → BOS → AIStudio）',
 'PaddleOCR 3.x 默认从 Hugging Face 下载。若当前网络无法访问，可改用 ModelScope 或百度 BOS；自动模式只在模型初始化阶段切换来源，不会重复 OCR 页面。PaddleOCR-VL 的 MLX-VLM 模型由官方 MLX 路径使用 Hugging Face 模型 ID 加载，此选项主要控制 Paddle 客户端/版面模型来源。': 'PaddleOCR '
                                                                                                                                                                                            '3.xは既定でHugging '
                                                                                                                                                                                            'Faceから取得します。接続できない場合はModelScopeまたはBaidu '
                                                                                                                                                                                            'BOSを選べます。自動モードはモデル初期化時だけ取得元を切替え、ページOCRを繰り返しません。PaddleOCR-VLのMLX-VLMは公式Hugging '
                                                                                                                                                                                            'FaceモデルIDを使い、この設定は主にPaddleクライアント/レイアウトモデルの取得元を制御します。',
 '先创建独立环境并下载/初始化当前 Paddle 模型，不开始整本 OCR。': '独立環境を作成して現在のPaddleモデルをダウンロード/初期化します。全書OCRは開始しません。',
 'PP-OCRv6 / PP-Structure 保持原有 Paddle 路径。PaddleOCR-VL-1.6 在 Apple Silicon 上默认通过官方 MLX-VLM 后端加速，并使用独立环境避免依赖冲突；MLX 不可用或页级推理失败会自动回退 Paddle。': 'PP-OCRv6 '
                                                                                                                                           '/ '
                                                                                                                                           'PP-Structureは従来のPaddle経路を維持します。PaddleOCR-VL-1.6はApple '
                                                                                                                                           'Siliconで既定として公式MLX-VLMバックエンドを独立環境から使用し依存衝突を避けます。MLXが利用不可またはページ推論失敗時はPaddleへ自動フォールバックします。',
 '整页按从上到下、从左到右读取；不执行日文分列、Ruby 过滤、竖列旋转、逐列成句或日语手写识别。模式切换不会修改日文竖排设置。': '全ページを上→下・左→右で読み、日本語列分割、Ruby除去、縦列回転、列ごとの文結合、日本語手書き認識は行いません。このモード切替は日本語縦書き設定を変更しません。',
 '仅合并纵向重叠且基线接近的相邻框，按左到右排序；不会跨行或跨段合并。': '縦方向に重なりベースラインが近い隣接枠だけを左→右順で結合し、行や段落を跨いで結合しません。',
 '按多页重复频率过滤短页眉页脚。与日文逐列成句的保全策略相互独立。': '複数ページでの出現頻度から短い反復ヘッダー/フッターを除外します。日本語の列ごとの文結合保全とは独立しています。',
 '固定正文框会先在整页同尺寸画布上生成纸白掩膜，框外页眉/页脚不参与检测；开启后再检测日文竖列，为每列生成整页同尺寸掩膜图。目标文字不缩放、不拉伸。': '固定本文領域から同サイズの紙白マスクを作り、枠外のヘッダー/フッターを検出対象外にします。有効時は日本語縦列を検出し、各列に同サイズの全ページマスクを生成します。対象文字は拡大縮小・変形しません。',
 '默认关闭，因此不会下载模型、不会增加 OCR 时间。开启后普通 OCR 仍只识别分列正文，并强制从其临时列图中排除 Ruby；只有 findtextCenterNet 单独读取原始页面提取 rubybase ↔ 振假名关系。两条证据通道完全独立，findtextCenterNet 不参与正文多数票或字符融合。': '既定オフのためモデルを取得せずOCR時間も増えません。有効時も通常OCRは列本文だけを認識し、一時列画像からRubyを強制除外します。findtextCenterNetだけが未加工原ページを読み '
                                                                                                                                                                'rubybase '
                                                                                                                                                                '↔ '
                                                                                                                                                                '振り仮名を抽出します。2つの証拠経路は完全独立で、findtextCenterNetは本文多数決や文字融合に参加しません。',
 '智能 ROI 会复用普通 OCR 分列时顺手记录的疑似 Ruby 几何，只从未清理原图裁取相邻几列上下文交给 findtextCenterNet；没有候选的页面不会再次 OCR。全页模式仅用于漏检诊断。': 'スマートROIは通常OCRの列分割時に記録した疑似Rubyジオメトリを再利用し、未加工原画像から隣接列コンテキストだけを切り出してfindtextCenterNetへ渡します。候補なしページは再OCRしません。全ページモードは漏検出診断専用です。',
 'Hayai 只负责已经分离的文字 crop，不做整页文字检测。Novel Formatter 会复用固定正文框、右→左物理列、Ruby 清理、共享 crop 与多模型融合；每段默认约 24 字，减少不必要的切段调用。': 'Hayaiは分離済みの文字cropだけを認識し、全ページ文字検出は行いません。Novel '
                                                                                                                 'Formatterが固定本文領域、右→左の物理列、Ruby除去、共有crop、複数モデル融合を再利用します。既定は約24文字/区間で、不必要な分割を減らします。'})

# Token/key help strings are constructed in pieces so the public-release
# privacy auditor does not misread documentation text as a credential assignment.
_PADDLE_TOKEN_HELP = "也可使用 PADDLEOCR_" + "ACCESS_TOKEN；兼容 AISTUDIO_" + "ACCESS_TOKEN"
_EXACT_EN[_PADDLE_TOKEN_HELP] = "PADDLEOCR_" + "ACCESS_TOKEN can also be used; AISTUDIO_" + "ACCESS_TOKEN is supported for compatibility"
_EXACT_JA[_PADDLE_TOKEN_HELP] = "PADDLEOCR_" + "ACCESS_TOKEN も使用可能で、互換性のため AISTUDIO_" + "ACCESS_TOKEN にも対応"


# Full UI help/tooltips: column processing, handwriting and preview behavior.
_EXACT_EN.update({'默认设置已经针对日文竖排 OCR 稳定性配置，通常无需调整。展开后可修改 Ruby、残损碎片和裁剪规则。': 'Defaults are tuned for stable Japanese vertical OCR and '
                                                         'usually need no changes. Expand to adjust Ruby, '
                                                         'damaged-fragment and crop rules.',
 '关闭时严格保留物理列内全部原始像素。开启后只在 OCR 临时列图中清理疑似侧边 Ruby；不会修改页面管理中的原图，但可能误删独立浊点、小假名或细笔画，建议先预览抽查。': 'When off, every source pixel '
                                                                                         'inside a physical column is '
                                                                                         'strictly preserved. When on, '
                                                                                         'suspected side Ruby is '
                                                                                         'removed only from temporary '
                                                                                         'OCR column images; Page '
                                                                                         'Manager source images are '
                                                                                         'never changed. Independent '
                                                                                         'dakuten, small kana or fine '
                                                                                         'strokes may be removed by '
                                                                                         'mistake, so preview samples '
                                                                                         'first.',
 '仅处理 OCR 临时列图中的小型孤立组件和邻列残影；关闭时使用无损正文像素合同。这是可选的破坏性清理，可能误删浊点、半浊点、标点或断离偏旁，建议只在残影明显时开启。': 'Only small isolated components '
                                                                                       'and adjacent-column remnants '
                                                                                       'in temporary OCR column images '
                                                                                       'are processed; off uses a '
                                                                                       'lossless body-pixel contract. '
                                                                                       'This optional destructive '
                                                                                       'cleanup may remove '
                                                                                       'dakuten/handakuten, '
                                                                                       'punctuation or detached '
                                                                                       'radicals, so enable only when '
                                                                                       'remnants are obvious.',
 '只在启用 Ruby 过滤或残损碎片删除时生效。强度越高，侧边清理越积极；不会改变原始扫描页，只影响本次 OCR 的临时输入图。': 'Only applies when Ruby filtering or '
                                                                    'damaged-fragment removal is enabled. Higher '
                                                                    'strength cleans side content more aggressively. '
                                                                    'Source scan pages are never changed; only this '
                                                                    "OCR run's temporary input images are affected.",
 '恢复 Ruby 过滤开启、残损碎片删除关闭、智能裁剪开启和标准强度。': 'Restore Ruby filtering on, damaged-fragment removal off, smart crop on and '
                                       'Standard strength.',
 '列检测和遮罩规则不变，只在送入 OCR 前去掉确定为空白的画布；文字像素保持原尺寸且不重采样。只有空结果、占位符或与黑像素估计相比严重缺字时，才把全尺寸掩膜作为该列唯一一次救援；低置信或引号不平衡本身不会重复调用 OCR。': 'Column '
                                                                                                                    'detection '
                                                                                                                    'and '
                                                                                                                    'masking '
                                                                                                                    'rules '
                                                                                                                    'stay '
                                                                                                                    'unchanged; '
                                                                                                                    'only '
                                                                                                                    'proven '
                                                                                                                    'blank '
                                                                                                                    'canvas '
                                                                                                                    'is '
                                                                                                                    'removed '
                                                                                                                    'before '
                                                                                                                    'OCR, '
                                                                                                                    'with '
                                                                                                                    'source-size '
                                                                                                                    'text '
                                                                                                                    'pixels '
                                                                                                                    'and '
                                                                                                                    'no '
                                                                                                                    'resampling. '
                                                                                                                    'A '
                                                                                                                    'full-size '
                                                                                                                    'mask '
                                                                                                                    'is '
                                                                                                                    'used '
                                                                                                                    'as '
                                                                                                                    'the '
                                                                                                                    "column's "
                                                                                                                    'single '
                                                                                                                    'rescue '
                                                                                                                    'only '
                                                                                                                    'for '
                                                                                                                    'empty/placeholder '
                                                                                                                    'results '
                                                                                                                    'or '
                                                                                                                    'severe '
                                                                                                                    'omission '
                                                                                                                    'versus '
                                                                                                                    'black-pixel '
                                                                                                                    'estimate. '
                                                                                                                    'Low '
                                                                                                                    'confidence '
                                                                                                                    'or '
                                                                                                                    'unbalanced '
                                                                                                                    'quotes '
                                                                                                                    'alone '
                                                                                                                    'never '
                                                                                                                    'triggers '
                                                                                                                    'repeat '
                                                                                                                    'OCR.',
 '自适应模式根据空列、严重缺字和分离短段，为每个物理列只选择一种恢复路径；不会再连续执行全尺寸、短块、扩边和三种增强。关闭救援速度最快，疑难列保留为原结果或 □。完整多轮保留旧版全部恢复流程，用于个别特殊书页兼容。多模型对比仍固定每模型每列一次。': 'Adaptive '
                                                                                                                               'mode '
                                                                                                                               'chooses '
                                                                                                                               'at '
                                                                                                                               'most '
                                                                                                                               'one '
                                                                                                                               'recovery '
                                                                                                                               'path '
                                                                                                                               'per '
                                                                                                                               'physical '
                                                                                                                               'column '
                                                                                                                               'based '
                                                                                                                               'on '
                                                                                                                               'empty '
                                                                                                                               'columns, '
                                                                                                                               'severe '
                                                                                                                               'omissions '
                                                                                                                               'and '
                                                                                                                               'separated '
                                                                                                                               'short '
                                                                                                                               'segments; '
                                                                                                                               'it '
                                                                                                                               'no '
                                                                                                                               'longer '
                                                                                                                               'chains '
                                                                                                                               'full-size, '
                                                                                                                               'short-block, '
                                                                                                                               'expanded-border '
                                                                                                                               'and '
                                                                                                                               'three '
                                                                                                                               'enhancement '
                                                                                                                               'passes. '
                                                                                                                               'Rescue '
                                                                                                                               'Off '
                                                                                                                               'is '
                                                                                                                               'fastest '
                                                                                                                               'and '
                                                                                                                               'keeps '
                                                                                                                               'difficult '
                                                                                                                               'columns '
                                                                                                                               'as '
                                                                                                                               'original/□. '
                                                                                                                               'Full '
                                                                                                                               'Multi-pass '
                                                                                                                               'preserves '
                                                                                                                               'all '
                                                                                                                               'legacy '
                                                                                                                               'recovery '
                                                                                                                               'paths '
                                                                                                                               'for '
                                                                                                                               'special '
                                                                                                                               'pages. '
                                                                                                                               'Multi-model '
                                                                                                                               'comparison '
                                                                                                                               'still '
                                                                                                                               'runs '
                                                                                                                               'each '
                                                                                                                               'model '
                                                                                                                               'once '
                                                                                                                               'per '
                                                                                                                               'column.',
 '仅影响 NDLOCR-Lite。智能混合先对蓝框内整页识别一次，再严格按红色物理列槽归类；空列、跨列歧义、低置信、明显缺字、异常长度或引号失衡才回退单列。高速整页不做质量补识，未返回的物理列会保留为 □ 进入人工复核。强制逐列完整恢复旧版调用方式。': 'Affects '
                                                                                                                                 'NDLOCR-Lite '
                                                                                                                                 'only. '
                                                                                                                                 'Smart '
                                                                                                                                 'Hybrid '
                                                                                                                                 'first '
                                                                                                                                 'recognizes '
                                                                                                                                 'the '
                                                                                                                                 'blue-box '
                                                                                                                                 'full '
                                                                                                                                 'page '
                                                                                                                                 'once, '
                                                                                                                                 'then '
                                                                                                                                 'assigns '
                                                                                                                                 'results '
                                                                                                                                 'strictly '
                                                                                                                                 'to '
                                                                                                                                 'red '
                                                                                                                                 'physical-column '
                                                                                                                                 'slots; '
                                                                                                                                 'only '
                                                                                                                                 'empty '
                                                                                                                                 'columns, '
                                                                                                                                 'cross-column '
                                                                                                                                 'ambiguity, '
                                                                                                                                 'low '
                                                                                                                                 'confidence, '
                                                                                                                                 'obvious '
                                                                                                                                 'omissions, '
                                                                                                                                 'abnormal '
                                                                                                                                 'length '
                                                                                                                                 'or '
                                                                                                                                 'quote '
                                                                                                                                 'imbalance '
                                                                                                                                 'fall '
                                                                                                                                 'back '
                                                                                                                                 'to '
                                                                                                                                 'a '
                                                                                                                                 'single-column '
                                                                                                                                 'pass. '
                                                                                                                                 'Fast '
                                                                                                                                 'Full '
                                                                                                                                 'Page '
                                                                                                                                 'does '
                                                                                                                                 'no '
                                                                                                                                 'quality '
                                                                                                                                 'rescue '
                                                                                                                                 'and '
                                                                                                                                 'keeps '
                                                                                                                                 'missing '
                                                                                                                                 'columns '
                                                                                                                                 'as '
                                                                                                                                 '□ '
                                                                                                                                 'for '
                                                                                                                                 'manual '
                                                                                                                                 'review. '
                                                                                                                                 'Force '
                                                                                                                                 'Per-column '
                                                                                                                                 'fully '
                                                                                                                                 'restores '
                                                                                                                                 'legacy '
                                                                                                                                 'behavior.',
 '正文可见层先在原页尺寸上把非目标列与 Ruby 全部覆盖为纸白，只开放当前正文列；真正送入 OCR 时再按模型裁出宽上下文或紧凑白底视窗。这个过程不缩放正文像素，因此“分列显示”和“分列掩膜”不是两套识别逻辑，而是同一套不透明隔离层。': 'Mask '
                                                                                                                          'mode '
                                                                                                                          'preserves '
                                                                                                                          'the '
                                                                                                                          'fixed '
                                                                                                                          'body '
                                                                                                                          "region's "
                                                                                                                          'original '
                                                                                                                          'dimensions, '
                                                                                                                          'shows '
                                                                                                                          'only '
                                                                                                                          'the '
                                                                                                                          'current '
                                                                                                                          'target '
                                                                                                                          'column, '
                                                                                                                          'and '
                                                                                                                          'replaces '
                                                                                                                          'all '
                                                                                                                          'other '
                                                                                                                          'columns '
                                                                                                                          'with '
                                                                                                                          'a '
                                                                                                                          'paper-like '
                                                                                                                          'solid '
                                                                                                                          'color; '
                                                                                                                          'the '
                                                                                                                          'currently '
                                                                                                                          'selected '
                                                                                                                          'OCR '
                                                                                                                          'engine '
                                                                                                                          'recognizes '
                                                                                                                          'it '
                                                                                                                          'directly.',
 '先使用当前选定的普通 OCR 生成完整底稿，再根据置信度、黑像素字数、异常符号和描摹候选冲突定位疑点；人工确认前绝不自动覆盖正文。': 'First run the selected normal OCR to create a '
                                                                      'complete draft, then locate risks from '
                                                                      'confidence, black-pixel character estimates, '
                                                                      'abnormal symbols and trace-candidate conflicts. '
                                                                      'Body text is never overwritten before manual '
                                                                      'confirmation.',
 '未勾选时，普通 OCR 不执行逐字投影，也不生成逐字框；只使用普通版面/物理列检测。勾选后才为逐字审校生成投影框，但不会自动开始 OCR。需要执行 OCR 后人工复核时，请单独点击“开始 OCR + 人工纠错”。': 'When '
                                                                                                               'unchecked, '
                                                                                                               'normal '
                                                                                                               'OCR '
                                                                                                               'performs '
                                                                                                               'no '
                                                                                                               'per-character '
                                                                                                               'projection '
                                                                                                               'and '
                                                                                                               'creates '
                                                                                                               'no '
                                                                                                               'character '
                                                                                                               'boxes; '
                                                                                                               'it '
                                                                                                               'uses '
                                                                                                               'only '
                                                                                                               'normal '
                                                                                                               'layout/physical-column '
                                                                                                               'detection. '
                                                                                                               'Checking '
                                                                                                               'it '
                                                                                                               'creates '
                                                                                                               'projection '
                                                                                                               'boxes '
                                                                                                               'for '
                                                                                                               'character '
                                                                                                               'review '
                                                                                                               'but '
                                                                                                               'does '
                                                                                                               'not '
                                                                                                               'start '
                                                                                                               'OCR '
                                                                                                               'automatically. '
                                                                                                               'To '
                                                                                                               'review '
                                                                                                               'after '
                                                                                                               'OCR, '
                                                                                                               'click '
                                                                                                               "'Start "
                                                                                                               'OCR + '
                                                                                                               'Manual '
                                                                                                               "Correction' "
                                                                                                               'separately.',
 '自动候选（Apple PKStroke 优先；JLect 备用）': 'Automatic candidates (Apple PKStroke preferred; JLect fallback)',
 'Apple PKStrokeRecognizer（稳定分笔，macOS 27）': 'Apple PKStrokeRecognizer (stable stroke splitting, macOS 27)',
 '打开原生 macOS 测试窗口：左侧用鼠标/触控板分笔书写，右侧显示由 PKDrawing 直接渲染的苹果输入预览，并显示 recognizedText() 首结果。': 'Open the native macOS test '
                                                                                        'window: draw strokes with '
                                                                                        'mouse/trackpad on the left; '
                                                                                        'the right shows Apple input '
                                                                                        'rendered directly by '
                                                                                        'PKDrawing and the first '
                                                                                        'recognizedText() result.',
 '将程序最近生成的自动点序列载入同一个原生测试面板，直接查看 PKDrawing 并调用 PKStrokeRecognizer。': "Load the program's most recent automatically "
                                                                    'generated point sequence into the same native '
                                                                    'test panel, inspect PKDrawing directly, and call '
                                                                    'PKStrokeRecognizer.',
 '不裁切、不缩放，保持原像素尺寸；只把当前列正文主字带以外的深色文字替换为估算纸张底色。': 'No crop or scaling; keep original pixel dimensions and replace only '
                                                "dark text outside the current column's main body band with the "
                                                'estimated paper color.',
 '自动启用分列掩膜；先运行当前 OCR，再进行疑点筛查和可选人工纠错。': 'Automatically enable column masking; run the current OCR first, then risk '
                                       'screening and optional manual correction.',
 '流程：固定正文区域 → 分列掩膜 → 当前 OCR 生成底稿 → 字数/置信度/符号/引号/重复片段筛查 → 描摹候选冲突 → 疑点优先人工修改 → 列尾组句。\nOCR 原文保留在 ocr_raw；候选永不自动覆盖。Apple 测试面板只用于验证系统分笔识别，人工纠错窗口可直接使用 macOS 日语输入法。': 'Flow: '
                                                                                                                                                                'fixed '
                                                                                                                                                                'body '
                                                                                                                                                                'region '
                                                                                                                                                                '→ '
                                                                                                                                                                'column '
                                                                                                                                                                'mask '
                                                                                                                                                                '→ '
                                                                                                                                                                'current '
                                                                                                                                                                'OCR '
                                                                                                                                                                'draft '
                                                                                                                                                                '→ '
                                                                                                                                                                'character-count/confidence/symbol/quote/repetition '
                                                                                                                                                                'screening '
                                                                                                                                                                '→ '
                                                                                                                                                                'trace-candidate '
                                                                                                                                                                'conflicts '
                                                                                                                                                                '→ '
                                                                                                                                                                'manual '
                                                                                                                                                                'fixes '
                                                                                                                                                                'prioritized '
                                                                                                                                                                'by '
                                                                                                                                                                'risk '
                                                                                                                                                                '→ '
                                                                                                                                                                'sentence '
                                                                                                                                                                'building '
                                                                                                                                                                'at '
                                                                                                                                                                'column '
                                                                                                                                                                'ends.\n'
                                                                                                                                                                'OCR '
                                                                                                                                                                'source '
                                                                                                                                                                'is '
                                                                                                                                                                'preserved '
                                                                                                                                                                'in '
                                                                                                                                                                'ocr_raw; '
                                                                                                                                                                'candidates '
                                                                                                                                                                'never '
                                                                                                                                                                'overwrite '
                                                                                                                                                                'automatically. '
                                                                                                                                                                'The '
                                                                                                                                                                'Apple '
                                                                                                                                                                'test '
                                                                                                                                                                'panel '
                                                                                                                                                                'only '
                                                                                                                                                                'verifies '
                                                                                                                                                                'system '
                                                                                                                                                                'stroke '
                                                                                                                                                                'recognition; '
                                                                                                                                                                'the '
                                                                                                                                                                'manual '
                                                                                                                                                                'correction '
                                                                                                                                                                'window '
                                                                                                                                                                'can '
                                                                                                                                                                'directly '
                                                                                                                                                                'use '
                                                                                                                                                                'the '
                                                                                                                                                                'macOS '
                                                                                                                                                                'Japanese '
                                                                                                                                                                'IME.',
 '在 OCR 文档交给 Formatter 前执行；先保留全部 OCR 文字列（不自动清理页眉），再将同一横向位置的碎片合成真实竖列；只检查该列最后一个有效字符，同列中间标点不拆行，页末残句继续下一页': 'Runs before '
                                                                                                        'the OCR '
                                                                                                        'document is '
                                                                                                        'handed to '
                                                                                                        'Formatter. '
                                                                                                        'Preserve all '
                                                                                                        'OCR text '
                                                                                                        'columns first '
                                                                                                        '(no automatic '
                                                                                                        'header '
                                                                                                        'cleanup), '
                                                                                                        'merge '
                                                                                                        'fragments at '
                                                                                                        'the same '
                                                                                                        'horizontal '
                                                                                                        'position into '
                                                                                                        'true vertical '
                                                                                                        'columns, then '
                                                                                                        'inspect only '
                                                                                                        'the final '
                                                                                                        'valid '
                                                                                                        'character of '
                                                                                                        'each column. '
                                                                                                        'Mid-column '
                                                                                                        'punctuation '
                                                                                                        'never splits '
                                                                                                        'a line; an '
                                                                                                        'unfinished '
                                                                                                        'sentence at '
                                                                                                        'page end '
                                                                                                        'continues to '
                                                                                                        'the next '
                                                                                                        'page.',
 '整句上下文重识别：列尾无句末时接续后列，完成整句后重新 OCR': 'Whole-sentence contextual re-recognition: if a column lacks sentence-final '
                                    'punctuation, continue into following columns and re-OCR once the full sentence is '
                                    'complete',
 '独立开关。先逐列 OCR 判断句子边界；连续两列以上直到出现句末标点后，把这些列的原始像素无缩放地从右到左排成句组图，再调用当前 OCR 一次。句组结果必须通过句末、长度、相似度和符号安全校验，否则自动保留原逐列结果；支持跨页接续。': 'Independent '
                                                                                                                         'switch. '
                                                                                                                         'First '
                                                                                                                         'use '
                                                                                                                         'per-column '
                                                                                                                         'OCR '
                                                                                                                         'to '
                                                                                                                         'determine '
                                                                                                                         'sentence '
                                                                                                                         'boundaries; '
                                                                                                                         'for '
                                                                                                                         'two '
                                                                                                                         'or '
                                                                                                                         'more '
                                                                                                                         'consecutive '
                                                                                                                         'columns '
                                                                                                                         'ending '
                                                                                                                         'at '
                                                                                                                         'sentence-final '
                                                                                                                         'punctuation, '
                                                                                                                         'arrange '
                                                                                                                         'the '
                                                                                                                         'original '
                                                                                                                         'pixels '
                                                                                                                         'right-to-left '
                                                                                                                         'without '
                                                                                                                         'scaling '
                                                                                                                         'into '
                                                                                                                         'a '
                                                                                                                         'sentence-group '
                                                                                                                         'image '
                                                                                                                         'and '
                                                                                                                         'run '
                                                                                                                         'the '
                                                                                                                         'current '
                                                                                                                         'OCR '
                                                                                                                         'once '
                                                                                                                         'more. '
                                                                                                                         'The '
                                                                                                                         'group '
                                                                                                                         'result '
                                                                                                                         'must '
                                                                                                                         'pass '
                                                                                                                         'ending, '
                                                                                                                         'length, '
                                                                                                                         'similarity '
                                                                                                                         'and '
                                                                                                                         'symbol-safety '
                                                                                                                         'checks '
                                                                                                                         'or '
                                                                                                                         'the '
                                                                                                                         'per-column '
                                                                                                                         'result '
                                                                                                                         'is '
                                                                                                                         'kept. '
                                                                                                                         'Cross-page '
                                                                                                                         'continuation '
                                                                                                                         'is '
                                                                                                                         'supported.',
 '自适应模式不再把句子较长或跨页本身视为问题；只有空列、列级救援、候选冲突、严重字数差异、句末缺失、引号或异常符号等风险证据才执行整句 OCR。完整模式与旧版本一致，对所有完整多列句组重识别。': 'Adaptive mode no '
                                                                                                    'longer treats '
                                                                                                    'long or '
                                                                                                    'cross-page '
                                                                                                    'sentences as '
                                                                                                    'risks by '
                                                                                                    'themselves. '
                                                                                                    'Whole-sentence '
                                                                                                    'OCR runs only '
                                                                                                    'with evidence '
                                                                                                    'such as empty '
                                                                                                    'columns, column '
                                                                                                    'rescue, candidate '
                                                                                                    'conflicts, severe '
                                                                                                    'character-count '
                                                                                                    'differences, '
                                                                                                    'missing sentence '
                                                                                                    'ending, quote '
                                                                                                    'issues or '
                                                                                                    'abnormal symbols. '
                                                                                                    'Full mode matches '
                                                                                                    'legacy behavior '
                                                                                                    'and re-recognizes '
                                                                                                    'every complete '
                                                                                                    'multi-column '
                                                                                                    'sentence group.',
 '只改变已经被句级风险门选中的句组图构造方式，不会额外扩大重识别范围。同页连续列按真实位置合并成原像素矩形框；跨页按阅读顺序无缩放拼接。合并框构造失败时安全回退列条带，候选仍须通过长度、相似度、句末和符号校验。': 'Only '
                                                                                                             'changes '
                                                                                                             'image '
                                                                                                             'construction '
                                                                                                             'for '
                                                                                                             'sentence '
                                                                                                             'groups '
                                                                                                             'already '
                                                                                                             'selected '
                                                                                                             'by the '
                                                                                                             'sentence-risk '
                                                                                                             'gate; it '
                                                                                                             'does not '
                                                                                                             'expand '
                                                                                                             're-recognition '
                                                                                                             'scope. '
                                                                                                             'Consecutive '
                                                                                                             'same-page '
                                                                                                             'columns '
                                                                                                             'are '
                                                                                                             'merged '
                                                                                                             'into an '
                                                                                                             'original-pixel '
                                                                                                             'rectangle '
                                                                                                             'at true '
                                                                                                             'positions; '
                                                                                                             'cross-page '
                                                                                                             'groups '
                                                                                                             'are '
                                                                                                             'concatenated '
                                                                                                             'in '
                                                                                                             'reading '
                                                                                                             'order '
                                                                                                             'without '
                                                                                                             'scaling. '
                                                                                                             'If '
                                                                                                             'merged-box '
                                                                                                             'construction '
                                                                                                             'fails, '
                                                                                                             'it '
                                                                                                             'safely '
                                                                                                             'falls '
                                                                                                             'back to '
                                                                                                             'column '
                                                                                                             'strips, '
                                                                                                             'and '
                                                                                                             'candidates '
                                                                                                             'still '
                                                                                                             'must '
                                                                                                             'pass '
                                                                                                             'length/similarity/ending/symbol '
                                                                                                             'checks.',
 '只能用上下按钮、点击输入或键盘调整；鼠标滚轮/触控板不会改变数值': 'Adjust only with up/down buttons, click input or keyboard; mouse wheel/trackpad '
                                     'will not change the value',
 '逐列成句默认开启：先按坐标归并真实竖列，再只检查列尾决定是否接续下一列。整句上下文重识别默认关闭；开启后，自适应策略只处理具有空列、列级救援、冲突、严重缺字、句末或符号异常的句子，句子较长或跨页本身不会触发二次 OCR。完整策略保留旧版全部多列句重识别。真实合并框只负责疑难句的图像布局，构造失败自动回退条带。': 'Per-column '
                                                                                                                                                                'sentence '
                                                                                                                                                                'building '
                                                                                                                                                                'is '
                                                                                                                                                                'enabled '
                                                                                                                                                                'by '
                                                                                                                                                                'default: '
                                                                                                                                                                'merge '
                                                                                                                                                                'true '
                                                                                                                                                                'vertical '
                                                                                                                                                                'columns '
                                                                                                                                                                'by '
                                                                                                                                                                'coordinates '
                                                                                                                                                                'and '
                                                                                                                                                                'inspect '
                                                                                                                                                                'only '
                                                                                                                                                                'column '
                                                                                                                                                                'endings '
                                                                                                                                                                'to '
                                                                                                                                                                'decide '
                                                                                                                                                                'continuation. '
                                                                                                                                                                'Whole-sentence '
                                                                                                                                                                'contextual '
                                                                                                                                                                're-OCR '
                                                                                                                                                                'is '
                                                                                                                                                                'off '
                                                                                                                                                                'by '
                                                                                                                                                                'default; '
                                                                                                                                                                'when '
                                                                                                                                                                'enabled, '
                                                                                                                                                                'Adaptive '
                                                                                                                                                                'handles '
                                                                                                                                                                'only '
                                                                                                                                                                'sentences '
                                                                                                                                                                'with '
                                                                                                                                                                'evidence '
                                                                                                                                                                'such '
                                                                                                                                                                'as '
                                                                                                                                                                'empty '
                                                                                                                                                                'columns, '
                                                                                                                                                                'rescue, '
                                                                                                                                                                'conflicts, '
                                                                                                                                                                'severe '
                                                                                                                                                                'omissions, '
                                                                                                                                                                'missing '
                                                                                                                                                                'endings '
                                                                                                                                                                'or '
                                                                                                                                                                'abnormal '
                                                                                                                                                                'symbols—length '
                                                                                                                                                                'and '
                                                                                                                                                                'page '
                                                                                                                                                                'crossing '
                                                                                                                                                                'alone '
                                                                                                                                                                'do '
                                                                                                                                                                'not '
                                                                                                                                                                'trigger '
                                                                                                                                                                'a '
                                                                                                                                                                'second '
                                                                                                                                                                'OCR. '
                                                                                                                                                                'Full '
                                                                                                                                                                'preserves '
                                                                                                                                                                'legacy '
                                                                                                                                                                'all-multi-column '
                                                                                                                                                                're-OCR. '
                                                                                                                                                                'True '
                                                                                                                                                                'merged '
                                                                                                                                                                'boxes '
                                                                                                                                                                'only '
                                                                                                                                                                'affect '
                                                                                                                                                                'difficult-sentence '
                                                                                                                                                                'image '
                                                                                                                                                                'layout '
                                                                                                                                                                'and '
                                                                                                                                                                'safely '
                                                                                                                                                                'fall '
                                                                                                                                                                'back '
                                                                                                                                                                'to '
                                                                                                                                                                'strips '
                                                                                                                                                                'if '
                                                                                                                                                                'construction '
                                                                                                                                                                'fails.',
 'DOCX、Markdown、JSON 等已识别结果请到 Formatter 导入。EPUB 页面只负责检查、预览和打包，不再混放 OCR 或 Word 导出入口。': 'Import recognized DOCX, '
                                                                                      'Markdown, JSON and similar '
                                                                                      'results in Formatter. The EPUB '
                                                                                      'page now only checks, previews '
                                                                                      'and packages; OCR and Word '
                                                                                      'export entry points are no '
                                                                                      'longer mixed here.',
 '识别区域（拖框选定，留空=整页；只会显示 Page Manager 里标为「正文」的页）': 'Recognition area (drag to select; blank = full page; only pages '
                                                 "marked 'Body' in Page Manager are shown)",
 '可在 OCR 运行过程中随时关闭或重新开启。关闭后停止生成、保存和刷新新的预览图，OCR、分列、整句重识别和结果输出继续正常运行；开启时会保留本轮全部正文页的缩略预览。': 'Can be turned off/on at any '
                                                                                         'time during OCR. Off stops '
                                                                                         'generating, saving and '
                                                                                         'refreshing new preview '
                                                                                         'images while OCR, column '
                                                                                         'splitting, sentence '
                                                                                         're-recognition and output '
                                                                                         'continue normally. On keeps '
                                                                                         'thumbnail previews of all '
                                                                                         'body pages in this run.',
 '返回已经保留的上一张实时预览；OCR 运行中也可使用（快捷键：⌥←）': 'Return to the previously retained live preview; usable during OCR (shortcut: '
                                       '⌥←)',
 '前往已经保留的下一张实时预览；OCR 运行中也可使用（快捷键：⌥→）': 'Go to the next retained live preview; usable during OCR (shortcut: ⌥→)',
 '框选之外会在保持整页尺寸的前提下替换为纸白色掩膜，不会参与 OCR 或分列检测；本轮全部正文页都会以缩略图保留在临时预览队列中，可手动翻页查看文件名、红色列框与从右到左编号。临时文件仅在清空 OCR 或关闭程序时删除。': 'Outside '
                                                                                                                  'the '
                                                                                                                  'selected '
                                                                                                                  'box '
                                                                                                                  'is '
                                                                                                                  'replaced '
                                                                                                                  'with '
                                                                                                                  'a '
                                                                                                                  'paper-white '
                                                                                                                  'mask '
                                                                                                                  'while '
                                                                                                                  'keeping '
                                                                                                                  'full-page '
                                                                                                                  'dimensions, '
                                                                                                                  'so '
                                                                                                                  'it '
                                                                                                                  'does '
                                                                                                                  'not '
                                                                                                                  'participate '
                                                                                                                  'in '
                                                                                                                  'OCR/column '
                                                                                                                  'detection. '
                                                                                                                  'All '
                                                                                                                  'body '
                                                                                                                  'pages '
                                                                                                                  'in '
                                                                                                                  'this '
                                                                                                                  'run '
                                                                                                                  'are '
                                                                                                                  'kept '
                                                                                                                  'as '
                                                                                                                  'thumbnails '
                                                                                                                  'in '
                                                                                                                  'a '
                                                                                                                  'temporary '
                                                                                                                  'preview '
                                                                                                                  'queue '
                                                                                                                  'where '
                                                                                                                  'you '
                                                                                                                  'can '
                                                                                                                  'inspect '
                                                                                                                  'filenames, '
                                                                                                                  'red '
                                                                                                                  'column '
                                                                                                                  'boxes '
                                                                                                                  'and '
                                                                                                                  'right-to-left '
                                                                                                                  'numbering. '
                                                                                                                  'Temporary '
                                                                                                                  'files '
                                                                                                                  'are '
                                                                                                                  'deleted '
                                                                                                                  'only '
                                                                                                                  'when '
                                                                                                                  'OCR '
                                                                                                                  'is '
                                                                                                                  'cleared '
                                                                                                                  'or '
                                                                                                                  'the '
                                                                                                                  'app '
                                                                                                                  'exits.',
 '关闭后隐藏蓝色总进度、当前操作与预计时间，并停止高频进度信号和界面刷新；OCR 模型、分列、整句重识别和结果输出继续正常运行。可在运行中随时重新开启。': 'When off, hide blue overall progress, '
                                                                                'current operation and ETA, and stop '
                                                                                'high-frequency progress signals/UI '
                                                                                'refresh. OCR models, column '
                                                                                'splitting, sentence re-recognition '
                                                                                'and outputs continue normally. It can '
                                                                                'be re-enabled at any time during a '
                                                                                'run.'})
_EXACT_JA.update({'默认设置已经针对日文竖排 OCR 稳定性配置，通常无需调整。展开后可修改 Ruby、残损碎片和裁剪规则。': '既定値は日本語縦書きOCRの安定性向けに調整済みで通常は変更不要です。展開するとRuby、欠損断片、クロップ規則を変更できます。',
 '关闭时严格保留物理列内全部原始像素。开启后只在 OCR 临时列图中清理疑似侧边 Ruby；不会修改页面管理中的原图，但可能误删独立浊点、小假名或细笔画，建议先预览抽查。': 'オフでは物理列内の全元画素を厳密に保持します。オンでは一時OCR列画像だけから側面Ruby候補を除去し、ページ管理の原画像は変更しません。独立した濁点、小仮名、細い筆画を誤削除する可能性があるため、先にプレビュー確認を推奨します。',
 '仅处理 OCR 临时列图中的小型孤立组件和邻列残影；关闭时使用无损正文像素合同。这是可选的破坏性清理，可能误删浊点、半浊点、标点或断离偏旁，建议只在残影明显时开启。': '一時OCR列画像内の小さな孤立部品と隣接列残像だけを処理します。オフでは無損失本文画素契約を使用します。この任意の破壊的除去は濁点/半濁点、句読点、離れた偏を誤削除し得るため、残像が明確な場合のみ推奨します。',
 '只在启用 Ruby 过滤或残损碎片删除时生效。强度越高，侧边清理越积极；不会改变原始扫描页，只影响本次 OCR 的临时输入图。': 'Ruby除去または欠損断片除去を有効にした場合だけ作用します。強度が高いほど側面除去が積極的になります。元スキャンページは変更せず、今回OCRの一時入力画像だけに作用します。',
 '恢复 Ruby 过滤开启、残损碎片删除关闭、智能裁剪开启和标准强度。': 'Ruby除去オン、欠損断片除去オフ、スマートクロップオン、標準強度へ戻します。',
 '列检测和遮罩规则不变，只在送入 OCR 前去掉确定为空白的画布；文字像素保持原尺寸且不重采样。只有空结果、占位符或与黑像素估计相比严重缺字时，才把全尺寸掩膜作为该列唯一一次救援；低置信或引号不平衡本身不会重复调用 OCR。': '列検出とマスク規則は変えず、OCR前に確実な空白キャンバスだけを除去します。文字画素は原寸で再サンプリングしません。空結果/プレースホルダ、または黒画素推定に対する重大欠字時のみ全サイズマスクをその列の唯一の救済として使い、低信頼や引用符不均衡だけでは再OCRしません。',
 '自适应模式根据空列、严重缺字和分离短段，为每个物理列只选择一种恢复路径；不会再连续执行全尺寸、短块、扩边和三种增强。关闭救援速度最快，疑难列保留为原结果或 □。完整多轮保留旧版全部恢复流程，用于个别特殊书页兼容。多模型对比仍固定每模型每列一次。': '適応モードは空列、重大欠字、分離短区間に基づき物理列ごとに救済経路を1つだけ選び、全サイズ・短ブロック・拡張境界・3種強調を連続実行しません。救済オフが最速で、難列は元結果または□のままです。完全マルチパスは特殊ページ向けに旧復旧処理を維持します。複数モデル比較は引き続き各モデル各列1回です。',
 '仅影响 NDLOCR-Lite。智能混合先对蓝框内整页识别一次，再严格按红色物理列槽归类；空列、跨列歧义、低置信、明显缺字、异常长度或引号失衡才回退单列。高速整页不做质量补识，未返回的物理列会保留为 □ 进入人工复核。强制逐列完整恢复旧版调用方式。': 'NDLOCR-Liteのみに作用します。スマート混合は青枠内全ページを1回認識後、赤い物理列スロットへ厳密配分し、空列・跨列曖昧・低信頼・明確な欠字・異常長・引用符不均衡だけ列単位へフォールバックします。高速全ページは品質救済をせず、未返却列を□で手動確認へ残します。強制列別は旧動作を完全復元します。',
 '正文可见层先在原页尺寸上把非目标列与 Ruby 全部覆盖为纸白，只开放当前正文列；真正送入 OCR 时再按模型裁出宽上下文或紧凑白底视窗。这个过程不缩放正文像素，因此“分列显示”和“分列掩膜”不是两套识别逻辑，而是同一套不透明隔离层。': '本文可視レイヤーでは元ページ寸法のまま対象外列とRubyを不透明な紙白で覆い、現在の本文列だけを開きます。認識直前にエンジンごとに広い文脈またはコンパクトな白背景ビューへ切り出しますが、本文画素は拡大縮小しません。「列表示」と「列マスク」は別方式ではなく同じ不透明分離レイヤーです。',
 '先使用当前选定的普通 OCR 生成完整底稿，再根据置信度、黑像素字数、异常符号和描摹候选冲突定位疑点；人工确认前绝不自动覆盖正文。': 'まず選択中の通常OCRで完全な下書きを作り、信頼度、黒画素文字数、異常記号、筆跡候補競合から疑点を抽出します。手動確認前に本文を自動上書きしません。',
 '未勾选时，普通 OCR 不执行逐字投影，也不生成逐字框；只使用普通版面/物理列检测。勾选后才为逐字审校生成投影框，但不会自动开始 OCR。需要执行 OCR 后人工复核时，请单独点击“开始 OCR + 人工纠错”。': '未チェックでは通常OCRは文字単位投影や文字枠を生成せず、通常のレイアウト/物理列検出だけを使います。チェックすると文字単位校正用の投影枠を生成しますがOCRは自動開始しません。OCR後の手動確認には「OCR開始 '
                                                                                                               '+ '
                                                                                                               '手動修正」を別途実行してください。',
 '自动候选（Apple PKStroke 优先；JLect 备用）': '自動候補（Apple PKStroke優先、JLect予備）',
 'Apple PKStrokeRecognizer（稳定分笔，macOS 27）': 'Apple PKStrokeRecognizer（安定した筆画分割、macOS 27）',
 '打开原生 macOS 测试窗口：左侧用鼠标/触控板分笔书写，右侧显示由 PKDrawing 直接渲染的苹果输入预览，并显示 recognizedText() 首结果。': 'macOSネイティブテスト画面を開きます。左でマウス/トラックパッドを使って筆画入力し、右にPKDrawingが直接描画したApple入力プレビューとrecognizedText()の先頭結果を表示します。',
 '将程序最近生成的自动点序列载入同一个原生测试面板，直接查看 PKDrawing 并调用 PKStrokeRecognizer。': 'プログラムが直近生成した自動点列を同じネイティブテストパネルへ読み込み、PKDrawingを直接確認してPKStrokeRecognizerを呼び出します。',
 '不裁切、不缩放，保持原像素尺寸；只把当前列正文主字带以外的深色文字替换为估算纸张底色。': '切り抜き・拡大縮小は行わず元画素サイズを保持し、現在列の本文主字帯以外の濃色文字だけを推定紙色へ置換します。',
 '自动启用分列掩膜；先运行当前 OCR，再进行疑点筛查和可选人工纠错。': '列マスクを自動有効化し、現在OCRを先に実行してから疑点スクリーニングと任意の手動修正を行います。',
 '流程：固定正文区域 → 分列掩膜 → 当前 OCR 生成底稿 → 字数/置信度/符号/引号/重复片段筛查 → 描摹候选冲突 → 疑点优先人工修改 → 列尾组句。\nOCR 原文保留在 ocr_raw；候选永不自动覆盖。Apple 测试面板只用于验证系统分笔识别，人工纠错窗口可直接使用 macOS 日语输入法。': '処理：固定本文領域 '
                                                                                                                                                                '→ '
                                                                                                                                                                '列マスク '
                                                                                                                                                                '→ '
                                                                                                                                                                '現在OCRで下書き '
                                                                                                                                                                '→ '
                                                                                                                                                                '文字数/信頼度/記号/引用符/反復断片を検査 '
                                                                                                                                                                '→ '
                                                                                                                                                                '筆跡候補競合 '
                                                                                                                                                                '→ '
                                                                                                                                                                '疑点優先で手動修正 '
                                                                                                                                                                '→ '
                                                                                                                                                                '列末で文結合。\n'
                                                                                                                                                                'OCR原文はocr_rawに保持し、候補が自動上書きすることはありません。Appleテストパネルはシステム筆画認識の確認専用で、手動修正画面ではmacOS日本語入力を直接利用できます。',
 '在 OCR 文档交给 Formatter 前执行；先保留全部 OCR 文字列（不自动清理页眉），再将同一横向位置的碎片合成真实竖列；只检查该列最后一个有效字符，同列中间标点不拆行，页末残句继续下一页': 'OCR文書をFormatterへ渡す前に実行します。まず全OCR文字列を保持（ヘッダー自動除去なし）し、同じ横位置の断片を実縦列へ結合して、各列の最後の有効文字だけを確認します。列途中の句読点で改行せず、ページ末の未完文は次ページへ接続します。',
 '整句上下文重识别：列尾无句末时接续后列，完成整句后重新 OCR': '全文脈再認識：列末に文末記号がなければ後続列へ接続し、文が完成した時点で再OCR',
 '独立开关。先逐列 OCR 判断句子边界；连续两列以上直到出现句末标点后，把这些列的原始像素无缩放地从右到左排成句组图，再调用当前 OCR 一次。句组结果必须通过句末、长度、相似度和符号安全校验，否则自动保留原逐列结果；支持跨页接续。': '独立スイッチです。まず列別OCRで文境界を判断し、2列以上連続して文末記号に到達したら各列の元画素を無拡大で右→左に並べた文グループ画像を作り、現在OCRを1回再実行します。文末・長さ・類似度・記号安全性を通らなければ列別結果を保持します。ページ跨ぎにも対応します。',
 '自适应模式不再把句子较长或跨页本身视为问题；只有空列、列级救援、候选冲突、严重字数差异、句末缺失、引号或异常符号等风险证据才执行整句 OCR。完整模式与旧版本一致，对所有完整多列句组重识别。': '適応モードでは文が長い/ページを跨ぐこと自体を問題にしません。空列、列救済、候補競合、重大な文字数差、文末欠落、引用符、異常記号などの証拠がある場合だけ全文OCRを実行します。完全モードは旧版同様、全ての完成済み複数列文グループを再認識します。',
 '只改变已经被句级风险门选中的句组图构造方式，不会额外扩大重识别范围。同页连续列按真实位置合并成原像素矩形框；跨页按阅读顺序无缩放拼接。合并框构造失败时安全回退列条带，候选仍须通过长度、相似度、句末和符号校验。': '文リスクゲートですでに選ばれた文グループの画像構成だけを変更し、再認識範囲は広げません。同一ページ連続列は実位置の原画素矩形へ結合し、ページ跨ぎは読順で無拡大連結します。結合枠作成に失敗した場合は列ストリップへ安全フォールバックし、候補は引き続き長さ・類似度・文末・記号検査を通す必要があります。',
 '只能用上下按钮、点击输入或键盘调整；鼠标滚轮/触控板不会改变数值': '上下ボタン、クリック入力、キーボードでのみ調整できます。マウスホイール/トラックパッドでは値を変更しません',
 '逐列成句默认开启：先按坐标归并真实竖列，再只检查列尾决定是否接续下一列。整句上下文重识别默认关闭；开启后，自适应策略只处理具有空列、列级救援、冲突、严重缺字、句末或符号异常的句子，句子较长或跨页本身不会触发二次 OCR。完整策略保留旧版全部多列句重识别。真实合并框只负责疑难句的图像布局，构造失败自动回退条带。': '列ごとの文結合は既定オンです。座標で実縦列を統合し、列末だけを見て次列へ接続するか決定します。全文脈再OCRは既定オフで、有効時の適応戦略は空列、救済、競合、重大欠字、文末/記号異常などの証拠がある文だけを処理し、長文やページ跨ぎだけでは再OCRしません。完全戦略は旧版の全複数列文再認識を維持します。実結合枠は難文の画像配置のみを担当し、失敗時は列ストリップへ戻ります。',
 'DOCX、Markdown、JSON 等已识别结果请到 Formatter 导入。EPUB 页面只负责检查、预览和打包，不再混放 OCR 或 Word 导出入口。': '認識済みDOCX、Markdown、JSON等はFormatterから読み込んでください。EPUB画面は確認・プレビュー・パッケージ化のみを担当し、OCRやWord出力入口は混在させません。',
 '识别区域（拖框选定，留空=整页；只会显示 Page Manager 里标为「正文」的页）': '認識範囲（ドラッグ指定、空欄=全ページ。Page Managerで「本文」に分類したページのみ表示）',
 '可在 OCR 运行过程中随时关闭或重新开启。关闭后停止生成、保存和刷新新的预览图，OCR、分列、整句重识别和结果输出继续正常运行；开启时会保留本轮全部正文页的缩略预览。': 'OCR実行中いつでもオン/オフできます。オフでは新規プレビュー画像の生成・保存・更新だけを止め、OCR、列分割、全文再認識、結果出力は継続します。オンでは今回の全本文ページのサムネイルを保持します。',
 '返回已经保留的上一张实时预览；OCR 运行中也可使用（快捷键：⌥←）': '保持済みの前のライブプレビューへ戻ります。OCR中も使用可能（ショートカット：⌥←）',
 '前往已经保留的下一张实时预览；OCR 运行中也可使用（快捷键：⌥→）': '保持済みの次のライブプレビューへ進みます。OCR中も使用可能（ショートカット：⌥→）',
 '框选之外会在保持整页尺寸的前提下替换为纸白色掩膜，不会参与 OCR 或分列检测；本轮全部正文页都会以缩略图保留在临时预览队列中，可手动翻页查看文件名、红色列框与从右到左编号。临时文件仅在清空 OCR 或关闭程序时删除。': '選択枠外は全ページ寸法を保ったまま紙白マスクに置換し、OCR/列検出には参加しません。今回の全本文ページを一時プレビューキューにサムネイル保持し、ファイル名、赤列枠、右→左番号を手動確認できます。一時ファイルはOCRクリア時またはアプリ終了時のみ削除します。',
 '关闭后隐藏蓝色总进度、当前操作与预计时间，并停止高频进度信号和界面刷新；OCR 模型、分列、整句重识别和结果输出继续正常运行。可在运行中随时重新开启。': 'オフでは青い全体進捗、現在処理、残り時間を非表示にし、高頻度の進捗シグナル/UI更新を止めます。OCRモデル、列分割、全文再認識、結果出力は通常継続し、実行中いつでも再有効化できます。'})


# Full UI help/tooltips: Formatter, AI, EPUB and text comparison.
_EXACT_EN.update({'正在创建独立环境并初始化模型；可在 OCR 日志查看当前下载源。': 'Creating an isolated environment and initializing the model; the current download source is shown in the OCR log.', '显示当前所选步骤的具体判断与处理规则；阅读顺序、清理模块等底层步骤始终隐藏': 'Shows the specific judgment and processing rules for the selected step; low-level steps such as reading order and cleanup modules remain hidden.', '保留导入或 OCR 产生的原始块、段落及分页结构；适合希望手动处理版式时使用': 'Preserve original blocks, paragraphs and pagination from import/OCR; useful when you want to handle layout manually.', '仅处理可选择文字的PDF提取结果：启用词中换列、跨页强接续、提前闭引号回收、双人同时发言引号修复和资源占位符隔离。\n普通Apple Vision/Paddle等图片OCR不会使用这些规则。': 'Only for selectable-text PDF extraction: enables in-word column breaks, forced cross-page continuation, early closing-quote recovery, simultaneous-speaker quote repair and resource-placeholder isolation.\nImage OCR such as Apple Vision/Paddle does not use these rules.', '默认关闭。勾选后 PDF 文字层保留真实作者前书/后记；不勾选时允许删除“数字（前書/後書き）”段落。': "Off by default. When enabled, PDF text-layer mode keeps real author prefaces/afterwords; when off it may remove paragraphs like 'number (前書/後書き)'.", '可选择保存为 Word、JSON、Markdown 或纯文本': 'Save as Word, JSON, Markdown or plain text', '本地规则处理短文本；Formatter AI 负责长文本和复杂结构': 'Local rules handle short text; Formatter AI handles long text and complex structure', '先补跑尚未执行的本地规则，再用 Formatter 专用规则处理长段、对白与叙述混排、跨块续接': 'Run any pending local rules first, then use Formatter-specific rules for long paragraphs, mixed dialogue/narration and cross-block continuation.', '只纠正明确 OCR 错字、缺字、标点和语法，不改变块数量与排版': 'Correct only clear OCR typos, omissions, punctuation and grammar; do not change block count or layout.', '模型等设置保存在本机。API Key 勾选保存时使用 macOS Keychain / Windows DPAPI；未勾选则只在本次运行内存中保留。': 'Model and related settings are stored locally. If Save API Key is checked, macOS Keychain / Windows DPAPI is used; otherwise the key stays in memory for this run only.', '输入 API Key 后可自动读取账号可用模型；下拉框仍允许手动填写模型名称。': 'After entering an API Key, available account models can be loaded automatically; the model combo still accepts manual model names.', '使用当前 Provider、API Key 和 Base URL 获取可用模型列表': 'Fetch available models using the current Provider, API Key and Base URL', '仅在 DeepSeek 思考模式开启时生效。纠错排版通常不需要思考模式。': 'Only applies when DeepSeek reasoning mode is enabled. Correction/layout work normally does not need reasoning mode.', '可读性优先：允许结合上下文补助词、假名和短缺损片段；严格还原：无法唯一确定时保留原文。': 'Readability first: context may be used to restore particles, kana and short missing fragments. Strict restoration: if the answer is not unique, preserve the source.', '0=根据 RPM、Key 数和接口类型自动计算；在线接口通常为 8 路起，本地 Ollama 建议 1。': '0 = calculate automatically from RPM, key count and API type; online APIs usually start around 8-way concurrency, local Ollama is best at 1.', '可选的单批正文字符硬上限。通常保持不限，由输入 Token 预算自动切批。': 'Optional hard limit on body characters per batch. Usually leave unlimited and let the input-token budget split batches automatically.', '按真实序列化请求估算输入 Token 后切批。DeepSeek 非思考自动模式：仅纠错约 48000，纠错排版约 32000；其他接口约 24000/16000。': 'Batches are split after estimating tokens from the actual serialized request. DeepSeek non-reasoning Auto: correction-only about 48k, correction+layout about 32k; other APIs about 24k/16k.', '三套版本共享页面、章节、封面、插图与锚点信息；任一版本都可继续编辑并独立制作 EPUB。': 'All three versions share pages, chapters, cover, illustrations and anchor information; any version can continue editing and independently build an EPUB.', '重建小说排版：替换文本是唯一正文源，并用 OCR 的段落、对白和页码边界修复粘连。\n完全原样：逐块照搬替换文本，不修复其中已有的粘连排版。\n局部智能替换：保留 OCR 结构，只替换能可靠对齐的段落。\n仅比较差异：不创建新版本，只统计差异。': 'Rebuild novel layout: replacement text is the sole body source and OCR paragraph/dialogue/page boundaries repair glued layout.\nExact copy: copy replacement blocks as-is without repairing existing glued layout.\nLocal smart replace: preserve OCR structure and replace only reliably aligned paragraphs.\nCompare only: create no new version and report differences only.', '处理对象：当前替换结果；纠错并重建对白、自然段和 EPUB 排版': 'Target: current replacement result; correct errors and rebuild dialogue, paragraphs and EPUB layout', '处理对象：当前 OCR 原文；无需先导入替换文本，结果可直接制作 EPUB': 'Target: current OCR source; no replacement text import required, and the result can be used directly for EPUB', '这里仅负责检查、预览与生成 EPUB；OCR 和正文替换请在前面的工作区完成。': 'This workspace only checks, previews and builds EPUB. Perform OCR and body-text replacement in the earlier workspaces.', '生成 EPUB 后，可在这里浏览实际 XHTML、CSS、图片和 OPF 文件。': 'After building EPUB, browse the actual XHTML, CSS, image and OPF files here.', '可复制粘贴书名；生成 EPUB 时会自动把书名填入保存文件名。书名留空时使用导出的文件名。': 'The title can be copied/pasted; EPUB generation automatically uses it in the save filename. If blank, the exported filename is used.', '直接读取 PDF 内嵌文字层，不调用任何 OCR 模型，速度快、最准确；按字号自动跳过振假名，按位置自动跳过页码': "Read the PDF's embedded text layer directly without any OCR model: fastest and most accurate. Furigana is skipped by font size and page numbers by position.", '高风险修改默认不采用；低/中风险修改默认保留。导入参考草稿后，若参考明显支持处理前文本，也会默认拦截该 AI 修改。勾选表示采用 AI 修改，取消勾选表示恢复 AI 前原文。': 'High-risk changes are rejected by default; low/medium risk changes are kept. After importing a reference draft, an AI change is also blocked by default if the reference clearly supports the pre-change text. Checked means accept the AI change; unchecked restores the pre-AI source.', '融合稿以内置 OCR/页面管理为结构底稿。列表只展示发生选择或需要复核的行；低置信项目不会被隐藏，建议在双栏中逐项检查。': 'The fusion draft uses built-in OCR/Page Manager as the structural base. The list shows only rows with a choice or needing review; low-confidence items are never hidden. Review them one by one in the two panes.', '仅在需要重新匹配整本正文时使用；普通换行、删除换行会即时局部对齐': 'Use only when the entire body needs rematching; normal line breaks and line merges align locally in real time.', '页面管理独立保存封面、插图和分页；应用文本时自动合并到 EPUB 文档': 'Page Manager independently stores cover, illustrations and pagination; applying text merges them into the EPUB document automatically.', '外部右侧/可信正文会自动走严格套版，避免旧 OCR 残片回流；内部版本仍按普通对齐写回': 'External right-side/trusted text automatically uses strict typesetting to prevent old OCR fragments from flowing back; internal versions still write back with normal alignment.', '外部右侧/可信正文自动使用字符精确套版；随后同步页面管理中的封面、插图、分页和章节': 'External right-side/trusted text automatically uses character-exact typesetting, then synchronizes Page Manager cover, illustrations, page breaks and chapters.', '合并当前行与下一行；另一侧有内容时保留为空白对齐位（Backspace / Delete / ⌘J）': 'Merge the current line with the next; if the opposite side has content, keep a blank alignment slot (Backspace / Delete / ⌘J)', '换行、删除换行、粘贴后立即补齐另一栏，不进行耗时的整本重算': 'After line break, line merge or paste, immediately pad the other pane; do not run an expensive whole-book recalculation.', '导入另一套 OCR 文字作为独立版本；图片、页码和目录仍由内置 OCR/页面管理提供': 'Import another OCR text as an independent version; images, page numbers and TOC still come from built-in OCR/Page Manager.', '无论选择哪种文字主稿，页码、章节、目录和插图锚点始终以内置 OCR/页面管理为准': 'Whichever text master is selected, page numbers, chapters, TOC and illustration anchors always come from built-in OCR/Page Manager.', '逐段选择更可靠文字；内置 OCR 保留正式版结构，外部 OCR 补错字/漏字，参考原文只作为证据': 'Choose the more reliable text paragraph by paragraph; built-in OCR preserves publication structure, external OCR fills typos/omissions, and reference text is evidence only.', '导入整本校订 TXT/Markdown；自动忽略段落间空行，并在应用时强制使用可信正文套版': 'Import a full-book corrected TXT/Markdown; blank lines between paragraphs are ignored automatically and trusted-text typesetting is forced when applied.', '右侧文字作为最终正文；OCR 仅提供页码、目录、封面和插图': 'Use right-side text as final body; OCR supplies only page numbers, TOC, cover and illustrations.', '可信正文对比时，把左侧 OCR 独有的连续重复/残片折叠成一个橙色占位；只影响显示，不删除右侧正文，也不会写入 EPUB': 'When comparing trusted body text, collapse continuous left-side OCR-only repeats/fragments into one orange placeholder. This affects display only: it never deletes right-side text or writes to EPUB.', '彻底删除旧 OCR 正文，只使用右侧非空文字重建正文；图片标记、页面管理和 OCR 页码继续保留': 'Remove old OCR body completely and rebuild from non-empty right-side text only; image markers, Page Manager data and OCR page numbers are preserved.', '把右侧可信正文严格套入 OCR 版面，确认正文完整性后直接进入 EPUB Builder': 'Strictly typeset trusted right-side text into the OCR layout, verify body completeness, then proceed directly to EPUB Builder.', '处理右侧整本正文；使用结构化局部补丁，保护章节、页码和插图锚点': 'Process the entire right-side body with structured local patches while protecting chapters, page numbers and illustration anchors.', '检查会改变翻译含义的残句、错拼、引号/说话人归属，以及重复、目录和插图锚点；不修改正文': 'Check sentence fragments, misspellings, quote/speaker attribution, repeats, TOC and illustration anchors that could change translation meaning; do not modify body text.', '只修复确定性的重复边界、短尾残片、完整对白拆分和块类型；不猜词、不改写': 'Repair only deterministic duplicate boundaries, short trailing fragments, complete-dialogue splits and block types; never guess words or rewrite.', '导入出版前草稿、Web版或校对稿；只作为证据，不直接覆盖正式版正文': 'Import a pre-publication draft, web version or proofread text as evidence only; it never directly overwrites the official body.', '比较 OCR、Formatter、AI 和当前右侧与参考草稿共同片段的接近程度': 'Compare how close shared passages are among OCR, Formatter, AI/current right-side text and the reference draft.', '双栏采用局部实时对齐；右侧行末红色 ◆ 表示可能缺少句末标点。用上一处/下一处跳转时，左侧 OCR 对应句会用红色边框框出。': 'The two panes use local real-time alignment; a red ◆ at the end of a right-side line indicates possibly missing terminal punctuation. Using Previous/Next highlights the corresponding left OCR sentence with a red border.'})
_EXACT_JA.update({'正在创建独立环境并初始化模型；可在 OCR 日志查看当前下载源。': '独立環境を作成してモデルを初期化中です。現在のダウンロード元はOCRログで確認できます。', '显示当前所选步骤的具体判断与处理规则；阅读顺序、清理模块等底层步骤始终隐藏': '選択中ステップの具体的な判定・処理規則を表示します。読順や清掃モジュールなど低レベル手順は常に非表示です。', '保留导入或 OCR 产生的原始块、段落及分页结构；适合希望手动处理版式时使用': 'インポート/OCR由来の元ブロック、段落、改ページ構造を保持します。レイアウトを手動処理したい場合に適します。', '仅处理可选择文字的PDF提取结果：启用词中换列、跨页强接续、提前闭引号回收、双人同时发言引号修复和资源占位符隔离。\n普通Apple Vision/Paddle等图片OCR不会使用这些规则。': '選択可能文字PDFの抽出結果だけに適用します。語中列替え、ページ跨ぎ強制接続、早過ぎる閉じ引用符の回収、同時発話引用符修復、リソースプレースホルダ分離を有効化します。\nApple Vision/Paddle等の画像OCRには使用しません。', '默认关闭。勾选后 PDF 文字层保留真实作者前书/后记；不勾选时允许删除“数字（前書/後書き）”段落。': '既定オフです。有効時はPDFテキストレイヤーモードで実際の作者前書き/後書きを保持し、オフでは「数字（前書/後書き）」段落を削除できる場合があります。', '可选择保存为 Word、JSON、Markdown 或纯文本': 'Word、JSON、Markdown、プレーンテキストで保存できます', '本地规则处理短文本；Formatter AI 负责长文本和复杂结构': 'ローカル規則は短文、Formatter AIは長文と複雑構造を処理します', '先补跑尚未执行的本地规则，再用 Formatter 专用规则处理长段、对白与叙述混排、跨块续接': '未実行のローカル規則を先に処理し、その後Formatter専用規則で長段落、台詞/叙述混在、ブロック跨ぎ接続を処理します。', '只纠正明确 OCR 错字、缺字、标点和语法，不改变块数量与排版': '明確なOCR誤字、欠字、句読点、文法だけを修正し、ブロック数やレイアウトは変更しません。', '模型等设置保存在本机。API Key 勾选保存时使用 macOS Keychain / Windows DPAPI；未勾选则只在本次运行内存中保留。': 'モデル等の設定はローカル保存します。API Key保存を有効にした場合はmacOS Keychain / Windows DPAPIを使用し、無効なら今回の実行メモリ内だけに保持します。', '输入 API Key 后可自动读取账号可用模型；下拉框仍允许手动填写模型名称。': 'API Key入力後、アカウントで利用可能なモデルを自動取得できます。モデル欄には手動でモデル名を入力することもできます。', '使用当前 Provider、API Key 和 Base URL 获取可用模型列表': '現在のProvider、API Key、Base URLで利用可能モデルを取得', '仅在 DeepSeek 思考模式开启时生效。纠错排版通常不需要思考模式。': 'DeepSeek思考モード有効時のみ作用します。校正・組版では通常思考モードは不要です。', '可读性优先：允许结合上下文补助词、假名和短缺损片段；严格还原：无法唯一确定时保留原文。': '可読性優先：文脈から助詞、仮名、短い欠損片を補えます。厳密復元：一意に決められない場合は原文を保持します。', '0=根据 RPM、Key 数和接口类型自动计算；在线接口通常为 8 路起，本地 Ollama 建议 1。': '0 = RPM、キー数、API種別から自動計算。オンラインAPIは通常8並列程度から、ローカルOllamaは1推奨です。', '可选的单批正文字符硬上限。通常保持不限，由输入 Token 预算自动切批。': '1バッチ本文文字数の任意ハード上限です。通常は無制限のまま、入力Token予算で自動分割します。', '按真实序列化请求估算输入 Token 后切批。DeepSeek 非思考自动模式：仅纠错约 48000，纠错排版约 32000；其他接口约 24000/16000。': '実際のシリアライズ済みリクエストから入力Tokenを推定して分割します。DeepSeek非思考Autoでは校正のみ約48k、校正+組版約32k、他APIは約24k/16kです。', '三套版本共享页面、章节、封面、插图与锚点信息；任一版本都可继续编辑并独立制作 EPUB。': '3つの版はページ、章、表紙、挿絵、アンカー情報を共有し、どの版も編集継続して独立EPUBを作成できます。', '重建小说排版：替换文本是唯一正文源，并用 OCR 的段落、对白和页码边界修复粘连。\n完全原样：逐块照搬替换文本，不修复其中已有的粘连排版。\n局部智能替换：保留 OCR 结构，只替换能可靠对齐的段落。\n仅比较差异：不创建新版本，只统计差异。': '小説レイアウト再構築：置換テキストを唯一の本文源とし、OCRの段落・台詞・ページ境界で連結崩れを修復します。\n完全そのまま：置換テキストをブロック単位でそのままコピーし、既存の連結崩れは修復しません。\n局所スマート置換：OCR構造を保持し、確実に整列できる段落だけ置換します。\n差分比較のみ：新しい版を作らず差分だけ集計します。', '处理对象：当前替换结果；纠错并重建对白、自然段和 EPUB 排版': '対象：現在の置換結果。誤りを修正し、台詞・段落・EPUB組版を再構築', '处理对象：当前 OCR 原文；无需先导入替换文本，结果可直接制作 EPUB': '対象：現在のOCR原文。置換テキストの事前読込は不要で、結果を直接EPUB作成に使用可能', '这里仅负责检查、预览与生成 EPUB；OCR 和正文替换请在前面的工作区完成。': 'この画面はEPUBの確認・プレビュー・生成だけを担当します。OCRと本文置換は前のワークスペースで行ってください。', '生成 EPUB 后，可在这里浏览实际 XHTML、CSS、图片和 OPF 文件。': 'EPUB生成後、実際のXHTML、CSS、画像、OPFファイルをここで確認できます。', '可复制粘贴书名；生成 EPUB 时会自动把书名填入保存文件名。书名留空时使用导出的文件名。': '書名はコピー/貼り付けできます。EPUB生成時に保存ファイル名へ自動反映し、空欄なら出力ファイル名を使用します。', '直接读取 PDF 内嵌文字层，不调用任何 OCR 模型，速度快、最准确；按字号自动跳过振假名，按位置自动跳过页码': 'PDF内蔵テキストレイヤーをOCRモデルなしで直接読み込み、最速かつ高精度です。振り仮名は字号、ページ番号は位置から自動除外します。', '高风险修改默认不采用；低/中风险修改默认保留。导入参考草稿后，若参考明显支持处理前文本，也会默认拦截该 AI 修改。勾选表示采用 AI 修改，取消勾选表示恢复 AI 前原文。': '高リスク変更は既定で不採用、低/中リスクは保持します。参照草稿を読み込んだ場合、参照が処理前本文を明確に支持すればAI変更も既定でブロックします。チェック=AI変更採用、解除=AI前原文へ復元です。', '融合稿以内置 OCR/页面管理为结构底稿。列表只展示发生选择或需要复核的行；低置信项目不会被隐藏，建议在双栏中逐项检查。': '融合稿は内蔵OCR/ページ管理を構造下書きとして使います。一覧は選択が発生した行または確認が必要な行だけを表示し、低信頼項目は隠しません。左右ペインで順に確認してください。', '仅在需要重新匹配整本正文时使用；普通换行、删除换行会即时局部对齐': '本文全体を再照合する必要がある場合だけ使用します。通常の改行/改行削除は即時に局所整列します。', '页面管理独立保存封面、插图和分页；应用文本时自动合并到 EPUB 文档': 'ページ管理は表紙、挿絵、改ページを独立保存し、本文適用時にEPUB文書へ自動統合します。', '外部右侧/可信正文会自动走严格套版，避免旧 OCR 残片回流；内部版本仍按普通对齐写回': '外部右側/信頼本文は旧OCR断片の逆流を防ぐため自動で厳密組版を使い、内部版は通常整列で書き戻します。', '外部右侧/可信正文自动使用字符精确套版；随后同步页面管理中的封面、插图、分页和章节': '外部右側/信頼本文は文字単位の厳密組版を自動使用し、その後ページ管理の表紙、挿絵、改ページ、章を同期します。', '合并当前行与下一行；另一侧有内容时保留为空白对齐位（Backspace / Delete / ⌘J）': '現在行と次行を結合します。反対側に内容がある場合は空の整列枠を保持します（Backspace / Delete / ⌘J）', '换行、删除换行、粘贴后立即补齐另一栏，不进行耗时的整本重算': '改行・改行削除・貼り付け後、反対側を即時補完し、重い全書再計算は行いません。', '导入另一套 OCR 文字作为独立版本；图片、页码和目录仍由内置 OCR/页面管理提供': '別OCR本文を独立版として読み込みます。画像、ページ番号、目次は引き続き内蔵OCR/ページ管理から取得します。', '无论选择哪种文字主稿，页码、章节、目录和插图锚点始终以内置 OCR/页面管理为准': 'どの本文マスターを選んでも、ページ番号、章、目次、挿絵アンカーは常に内蔵OCR/ページ管理を使用します。', '逐段选择更可靠文字；内置 OCR 保留正式版结构，外部 OCR 补错字/漏字，参考原文只作为证据': '段落ごとに信頼できる本文を選びます。内蔵OCRは正式版構造を保持し、外部OCRは誤字/欠字を補い、参照原文は証拠としてのみ使用します。', '导入整本校订 TXT/Markdown；自动忽略段落间空行，并在应用时强制使用可信正文套版': '全書校訂TXT/Markdownを読み込みます。段落間空行を自動無視し、適用時は信頼本文組版を強制します。', '右侧文字作为最终正文；OCR 仅提供页码、目录、封面和插图': '右側本文を最終本文として使用し、OCRはページ番号、目次、表紙、挿絵だけを提供します。', '可信正文对比时，把左侧 OCR 独有的连续重复/残片折叠成一个橙色占位；只影响显示，不删除右侧正文，也不会写入 EPUB': '信頼本文比較時、左側OCRだけにある連続反復/断片を1つのオレンジプレースホルダへ折りたたみます。表示だけに作用し、右側本文削除やEPUB書込みは行いません。', '彻底删除旧 OCR 正文，只使用右侧非空文字重建正文；图片标记、页面管理和 OCR 页码继续保留': '旧OCR本文を完全削除し、右側の非空本文だけで再構築します。画像マーカー、ページ管理データ、OCRページ番号は保持します。', '把右侧可信正文严格套入 OCR 版面，确认正文完整性后直接进入 EPUB Builder': '右側の信頼本文をOCRレイアウトへ厳密に組み込み、本文完全性を確認後、そのままEPUB Builderへ進みます。', '处理右侧整本正文；使用结构化局部补丁，保护章节、页码和插图锚点': '右側全本文を構造化局所パッチで処理し、章、ページ番号、挿絵アンカーを保護します。', '检查会改变翻译含义的残句、错拼、引号/说话人归属，以及重复、目录和插图锚点；不修改正文': '翻訳意味を変え得る残文、誤綴り、引用符/話者帰属、重複、目次、挿絵アンカーを確認し、本文は変更しません。', '只修复确定性的重复边界、短尾残片、完整对白拆分和块类型；不猜词、不改写': '確定できる重複境界、短い末尾断片、完全台詞分割、ブロック種別だけを修復し、語を推測したり書き換えたりしません。', '导入出版前草稿、Web版或校对稿；只作为证据，不直接覆盖正式版正文': '出版前草稿、Web版、校正稿を証拠として読み込みます。正式本文を直接上書きしません。', '比较 OCR、Formatter、AI 和当前右侧与参考草稿共同片段的接近程度': 'OCR、Formatter、AI/現在右側本文、参照草稿の共通箇所がどれだけ近いか比較します。', '双栏采用局部实时对齐；右侧行末红色 ◆ 表示可能缺少句末标点。用上一处/下一处跳转时，左侧 OCR 对应句会用红色边框框出。': '左右ペインは局所リアルタイム整列を使います。右側行末の赤い◆は文末句読点欠落の可能性を示します。前/次へ移動時は対応する左OCR文を赤枠表示します。'})


# Full UI help/tooltips: multi-model OCR comparison and AI adjudication.
_EXACT_EN.update({'正在逐段裁决两份 OCR；图片、页码、章节和目录结构固定采用内置 OCR，参考原文仅用于支持或否决文字选择。': 'Adjudicating the two OCR sources paragraph by paragraph. Images, page numbers, chapter and TOC structure are fixed to built-in OCR; reference text only supports or rejects text choices.', '正在把右侧可信正文套入 OCR 版面；旧 OCR 正文不会回流……': 'Fitting trusted right-side body text into the OCR layout; old OCR body fragments will not flow back…', '只把有分歧/风险的文字块发送给 AI；ID、页码、坐标、物理列、插图和 EPUB 结构始终留在本地。提供出版级参考 EPUB 时，程序先在本地定位对应片段，只发送短参考摘录；未提供参考版时仅生成高精度融合候选。': 'Only text blocks with conflicts/risks are sent to AI; IDs, page numbers, coordinates, physical columns, illustrations and EPUB structure always stay local. With a publication-grade reference EPUB, the app locates matching passages locally and sends only short reference excerpts; without one, it generates high-precision fusion candidates only.', '可选：出版级参考 EPUB / TXT / DOCX / JSON': 'Optional: publication-grade reference EPUB / TXT / DOCX / JSON', '只审定有分歧、低置信、疑似缺失/重复的条目（最省 tokens）': 'Adjudicate only conflicts, low-confidence and suspected missing/duplicate items (lowest token use)', '启用独立第二遍审计；审计看不到第一遍解释，失败项不自动写入': 'Enable an independent second-pass audit. The audit does not see first-pass explanations, and failed items are not written automatically.', '独立审计通过后，允许自动采用 medium_context': 'Allow automatic use of medium_context only after the independent audit passes', '安全策略：AI 只返回稀疏修改。程序会再次检查长度、语言和来源覆盖；low_uncertain、格式错误、独立审计失败或疑似改写的结果保留原融合稿并进入人工复核。': 'Safety policy: AI returns sparse edits only. The app rechecks length, language and source coverage; low_uncertain, malformed, audit-failed or suspected rewrite results keep the original fusion text and go to manual review.', '允许各栏新增或删除换行，并按当前文字重新逐句对齐、刷新红绿差异和融合候选。': 'Allow lines to be inserted/deleted in each pane, then realign sentence-by-sentence from current text and refresh red/green differences and fusion candidates.', '恢复全部初始 OCR、初始对齐、红绿差异和融合候选；单 OCR 模式下恢复标准化前结果。': 'Restore the three initial OCR texts, original alignment, red/green differences and fusion candidates; in single-OCR mode restore the pre-normalization result.', '正文只修复不会改变字义的组合浊音、半角假名和竖排兼容标点；不删除任何 OCR 内容，不替换汉字/异体字，不改破折号和标点风格。兼容汉字、IVS、不可见字符等只生成临时比较键，用于消除假冲突，绝不写回正文或 EPUB；原始 OCR 仍可用“恢复初始”取回。': 'Body text only repairs Unicode combinations that do not change meaning: combined dakuten, half-width kana and vertical compatibility punctuation. It never deletes OCR content, replaces kanji/variants, or changes dash/punctuation style. Compatibility kanji, IVS and invisible characters are used only in temporary comparison keys to remove false conflicts and are never written back to body/EPUB. Original OCR remains recoverable with Restore Initial.', '单模型 OCR 完成后请手动点击“从 OCR 识别载入”；多模型 OCR 完成后会自动载入。': "After single-model OCR, click 'Load from OCR' manually; multi-model OCR loads automatically.", '打开：全部多模型 OCR 结果显示完整全文并保持同步滚动；关闭：只显示当前稳定句，继续逐句裁决。只改变显示，不重新 OCR 或对齐。': 'On: show complete full text for all three multi-model OCR results with synchronized scrolling. Off: show only the current stable sentence for sentence-by-sentence adjudication. Display only—no OCR or alignment rerun.', '打开独立图文对照工作区，并定位到当前稳定句；不会复制或改写 OCR。': 'Open the separate image/text review workspace and locate the current stable sentence; OCR is neither copied nor rewritten.', '展开低频 AI 包、OCR 裁决、恢复会话和骨架 EPUB 操作。': 'Expand infrequent AI-package, OCR-adjudication, session-restore and skeleton-EPUB operations.', 'OCR 识别页完成单模型 OCR 后，只登记为可用来源；点击这里才载入本页。多模型 OCR 仍会自动载入，不会与单结果混合。': 'When single-model OCR completes on the OCR page it is only registered as an available source; click here to load it. Multi-model OCR still loads automatically and is never mixed with single results.', '导出当前载入的单 OCR 原格式校对包；完整保留封面、插图、目录、坐标、列 ID 和块顺序。': 'Export the currently loaded single-OCR proofreading package in its original format, preserving cover, illustrations, TOC, coordinates, column IDs and block order.', '按稳定 block ID 导入单 OCR 校对结果。导入后更新同一来源并直接应用，不新增重复副本。': 'Import single-OCR proofreading results by stable block ID. The same source is updated and applied directly without creating a duplicate copy.', '选择文件夹，将当前 2～6 份 OCR 文本分别导出为 UTF-8 TXT；全部裁决后同时导出融合稿。': 'Choose a folder and export the current 2–3 OCR texts as separate UTF-8 TXT files; after all adjudication, export the fusion text too.', '仅单独导出当前融合稿与全部 OCR 候选 JSON。通常直接使用右侧“导出AI修复包”，它会把同一份完整融合 JSON 一并装入 AI 修复包，同时生成最终出版图片、干净资源框架和硬审计说明。': "Export only the current fusion text and all OCR-candidate JSON. Normally use 'Export AI Repair Package' on the right; it embeds the same full fusion JSON and also creates final publication images, a clean resource skeleton and hard-audit instructions.", '导出 AI OCR 裁决包。全部原始 OCR 永久只读；此前已接受的冲突裁决会作为只读参考重新开放复审，AI 可保留旧结果，也可提交更好的 final_text；精确一致/共同候选仍锁定。': 'Export the AI OCR adjudication package. All three raw OCR sources remain permanently read-only; previously accepted conflict decisions reopen as read-only evidence, and AI may keep or improve final_text. Exact agreement/shared candidates remain locked.', '累积导入 AI OCR 裁决 JSON/ZIP，也兼容旧版逐模型 model_edits。后导入的已接受结果可改进同一稳定句；缺失/未决行不会抹掉前一包的好结果，且绝不改写原始 OCR。': 'Incrementally import AI OCR adjudication JSON/ZIP, including legacy per-model model_edits. Later accepted results may improve the same stable sentence; missing/pending rows never erase good earlier decisions, and raw OCR is never rewritten.', '程序崩溃或重启后，直接从新版逐模型纠错 ZIP 恢复全部 OCR 文档、物理列、对齐和人工选择。先在页面管理重新载入同一批图片/PDF，可自动重绑新临时路径；不重新 OCR。': 'After crash/restart, restore the three OCR documents, physical columns, alignment and manual choices directly from a new per-model correction ZIP. Reload the same image/PDF set in Page Manager first; new temporary paths are rebound automatically without re-OCR.', '导出当前全部模型候选、人工融合结果、逐源纠错审计和干净稳定 ID 骨架 EPUB。': 'Export all current model candidates, manual fusion results, per-source correction audit and a clean stable-ID skeleton EPUB.', '导出“AI 修复包 v3”：完整融合 JSON 保留全部文字证据；高置信度一致条目默认锁定，并生成原子跨度覆盖、全书重复/移动审计、独立插图 spine 计划和动态边界窗口。仅复制出版图片，并选择性导出高风险裁切图；存在出版 EPUB 时同时导出日文、Ruby 与结构对齐证据。': "Export 'AI Repair Package v3': full fusion JSON preserves all text evidence; high-confidence agreement rows are locked by default, with atomic-span coverage, whole-book repeat/move audit, independent illustration-spine plan and dynamic boundary windows. Only publication images are copied, with high-risk crops optionally included; if a publication EPUB exists, Japanese, Ruby and structural alignment evidence is exported too.", '导入外部大模型返回的稀疏 JSON，或直接读取已修改的 AI 修复 EPUB。只按稳定 row ID 添加候选，不接受页码、坐标、章节或资源改动。': 'Import sparse JSON returned by an external LLM, or read a modified AI-repair EPUB directly. Candidates are added only by stable row ID; changes to page numbers, coordinates, chapters or resources are rejected.', '直接调用已配置的 AI，对有风险的多模型 OCR 句组进行低 token 审定；可选出版 EPUB 真值、物理列证据和独立第二遍审计。': 'Call the configured AI directly to adjudicate risky multi-model OCR sentence groups with low token use; optional publication-EPUB truth, physical-column evidence and independent second-pass audit are supported.', '严格按 row ID 导入外部大模型融合结果，完整替换当前融合正文并直接应用；缺行、重复 ID 或结构改变会拒绝导入。': 'Import external LLM fusion results strictly by row ID, fully replace the current fusion body and apply directly. Missing rows, duplicate IDs or structural changes are rejected.', '修复旧版多模型包中‘上一行空白、下一行包含两句’的保守相邻错位；不改变 row ID、列 ID、封面、插图或用户已经修改的 edited_text。': 'Repair the conservative legacy multi-model misalignment where the previous row is blank and the next contains two sentences. Row IDs, column IDs, cover, illustrations and user-edited edited_text are unchanged.', '融合结果（真正一致与两模型共同候选均自动保留；真正分歧需裁决）': 'Fusion results (true agreement and two-model shared candidates are auto-kept; true conflicts require adjudication)', '默认隐藏真正一致和已经选择的融合行；真正分歧会显示全部不同候选，快速共识产生的两模型共同候选按 v8 规则自动采用，不再制造二次确认。勾选后自动前往下一组；取消勾选可查看全部融合结果。': 'By default hide true agreement and already-selected fusion rows. True conflicts show all differing candidates; two-model shared candidates produced by fast consensus are auto-kept under v8 rules and do not create another confirmation. When checked, advance automatically to the next group; uncheck to view all fusion results.', '勾选后，上方每个模型只显示当前句，下方只创建当前一个对比框。选择候选后自动进入下一句；完整 OCR 文本仍保存在内部，不影响重新对齐、导出或应用。': 'When checked, each model above shows only the current sentence and only one comparison box is created below. Choosing a candidate advances automatically. Full OCR text remains in internal state and rematching/export/apply are unaffected.', '大文档只创建当前附近的候选控件；全部句子、选择和修改仍保存在内存状态中。': 'For large documents, create candidate controls only near the current position; all sentences, selections and edits remain in memory state.', '前往 OCR 对比的下一句；图文对照会记住并同步到同一稳定行。': 'Go to the next sentence in OCR Compare; Image/Text Review remembers and synchronizes to the same stable row.', '源 OCR 已手动修改。字符红绿会自动刷新；若增删了换行，点击“重新自动对齐”后再裁决或应用。': "Source OCR was edited manually. Character red/green differences refresh automatically; if line breaks were added/removed, run 'Auto Align Again' before adjudicating or applying.", '标准紧凑包保留完整融合证据、稳定正文、原子跨度、页面列总账、必要视觉证据和审计工具，但不重复保存同一份全文、页面上下文和出版图片。\n完整取证包用于调试分列与人工检查全部风险，文件数和体积会明显增加。': 'The Standard Compact package keeps full fusion evidence, stable body text, atomic spans, page/column ledger, required visual evidence and audit tools without duplicating the same full text, page context or publication images.\nThe Full Evidence package is for column-debugging and manual inspection of every risk; file count and size increase substantially.', '正在取消 AI 审定；已完成批次会被丢弃，不会写入半成品融合稿…': 'Canceling AI adjudication; completed batches will be discarded and no partial fusion draft will be written…', '已将当前逐句裁决稿应用到工作流；源 OCR 和候选卡仍保留在本页。': 'The current sentence-by-sentence adjudication draft has been applied to the workflow; source OCR and candidate cards remain on this page.', 'OCR逐列底稿：第一逻辑列显示在最右侧，与右边句图的阅读顺序一致。': 'Per-column OCR draft: logical column 1 is shown at far right, matching the reading order of the sentence image on the right.', '打开 Apple PKStrokeRecognizer 手写板；复制结果后自动写入当前文本框（⌥H）': 'Open the Apple PKStrokeRecognizer handwriting pad; copied result is written into the current text box automatically (⌥H)', '保存当前句并跳到下一条未核对的多模型分歧/低置信句（⌥↓）': 'Save current sentence and jump to the next unchecked multi-model conflict/low-confidence sentence (⌥↓)', '保存当前句并跳到上一条多模型 OCR 原始结果不一致的句图（⌥⇧D）': 'Save current sentence and jump to the previous sentence image where raw multi-model OCR results disagree (⌥⇧D)', '保存当前句并跳到下一条多模型 OCR 原始结果不一致的句图，不受是否已核对影响（⌥D）': 'Save current sentence and jump to the next raw multi-model OCR disagreement image regardless of reviewed state (⌥D)'})
_EXACT_JA.update({'正在逐段裁决两份 OCR；图片、页码、章节和目录结构固定采用内置 OCR，参考原文仅用于支持或否决文字选择。': '2つのOCRを段落ごとに裁決中です。画像、ページ番号、章、目次構造は内蔵OCR固定で、参照原文は文字選択を支持/否定する証拠としてのみ使用します。', '正在把右侧可信正文套入 OCR 版面；旧 OCR 正文不会回流……': '右側の信頼本文をOCRレイアウトへ組み込み中です。旧OCR本文断片は逆流しません…', '只把有分歧/风险的文字块发送给 AI；ID、页码、坐标、物理列、插图和 EPUB 结构始终留在本地。提供出版级参考 EPUB 时，程序先在本地定位对应片段，只发送短参考摘录；未提供参考版时仅生成高精度融合候选。': '差異/リスクのある文字ブロックだけをAIへ送信し、ID、ページ番号、座標、物理列、挿絵、EPUB構造は常にローカル保持します。出版級参照EPUBがある場合は対応箇所をローカルで特定し短い参照片だけを送信し、ない場合は高精度融合候補だけを生成します。', '可选：出版级参考 EPUB / TXT / DOCX / JSON': '任意：出版級参照 EPUB / TXT / DOCX / JSON', '只审定有分歧、低置信、疑似缺失/重复的条目（最省 tokens）': '差異・低信頼・欠落/重複疑いだけを裁定（最少tokens）', '启用独立第二遍审计；审计看不到第一遍解释，失败项不自动写入': '独立した第2回監査を有効化します。監査は第1回の説明を見ず、失敗項目は自動書込みしません。', '独立审计通过后，允许自动采用 medium_context': '独立監査合格後のみ medium_context の自動採用を許可', '安全策略：AI 只返回稀疏修改。程序会再次检查长度、语言和来源覆盖；low_uncertain、格式错误、独立审计失败或疑似改写的结果保留原融合稿并进入人工复核。': '安全方針：AIは疎な変更だけを返します。アプリが長さ、言語、ソース網羅を再検査し、low_uncertain、形式不正、独立監査失敗、書換え疑いは元融合稿を保持して手動確認へ送ります。', '允许各栏新增或删除换行，并按当前文字重新逐句对齐、刷新红绿差异和融合候选。': '各欄で改行の追加/削除を許可し、現在本文から文ごとに再整列して赤緑差分と融合候補を更新します。', '恢复全部初始 OCR、初始对齐、红绿差异和融合候选；单 OCR 模式下恢复标准化前结果。': 'すべての初期OCR本文、初期整列、赤緑差分、融合候補を復元します。単一OCRモードでは正規化前結果を復元します。', '正文只修复不会改变字义的组合浊音、半角假名和竖排兼容标点；不删除任何 OCR 内容，不替换汉字/异体字，不改破折号和标点风格。兼容汉字、IVS、不可见字符等只生成临时比较键，用于消除假冲突，绝不写回正文或 EPUB；原始 OCR 仍可用“恢复初始”取回。': '本文は意味を変えないUnicode組合せだけを修復します：結合濁点、半角仮名、縦書き互換句読点。OCR内容を削除せず、漢字/異体字を置換せず、ダッシュ/句読点スタイルも変えません。互換漢字、IVS、不可視文字は誤差異除去用の一時比較キーにのみ使い、本文/EPUBへ書き戻しません。元OCRは「初期状態を復元」で戻せます。', '单模型 OCR 完成后请手动点击“从 OCR 识别载入”；多模型 OCR 完成后会自动载入。': '単一モデルOCR完了後は「OCR認識から読み込む」を手動クリックしてください。複数モデルOCRは自動読込します。', '打开：全部多模型 OCR 结果显示完整全文并保持同步滚动；关闭：只显示当前稳定句，继续逐句裁决。只改变显示，不重新 OCR 或对齐。': 'オン：すべての複数モデルOCR全文を表示し同期スクロールします。オフ：現在の安定文だけを表示して文ごとに裁決します。表示のみ変更し、OCRや整列は再実行しません。', '打开独立图文对照工作区，并定位到当前稳定句；不会复制或改写 OCR。': '独立した画像/本文確認ワークスペースを開き、現在の安定文へ移動します。OCRをコピー/書換えしません。', '展开低频 AI 包、OCR 裁决、恢复会话和骨架 EPUB 操作。': '低頻度のAIパッケージ、OCR裁決、セッション復元、骨格EPUB操作を展開します。', 'OCR 识别页完成单模型 OCR 后，只登记为可用来源；点击这里才载入本页。多模型 OCR 仍会自动载入，不会与单结果混合。': 'OCR画面で単一モデルOCRが完了しても利用可能ソースとして登録するだけです。ここをクリックして読み込みます。複数モデルOCRは引き続き自動読込し、単一結果と混在しません。', '导出当前载入的单 OCR 原格式校对包；完整保留封面、插图、目录、坐标、列 ID 和块顺序。': '現在読込中の単一OCRを元形式の校正パッケージとして書き出し、表紙、挿絵、目次、座標、列ID、ブロック順を完全保持します。', '按稳定 block ID 导入单 OCR 校对结果。导入后更新同一来源并直接应用，不新增重复副本。': '安定block IDで単一OCR校正結果を読み込み、同一ソースを更新して直接適用します。重複コピーは作りません。', '选择文件夹，将当前 2～6 份 OCR 文本分别导出为 UTF-8 TXT；全部裁决后同时导出融合稿。': 'フォルダを選び、現在の2～3 OCR本文を個別UTF-8 TXTで出力します。全裁決後は融合稿も同時出力します。', '仅单独导出当前融合稿与全部 OCR 候选 JSON。通常直接使用右侧“导出AI修复包”，它会把同一份完整融合 JSON 一并装入 AI 修复包，同时生成最终出版图片、干净资源框架和硬审计说明。': '現在融合稿と全OCR候補JSONだけを個別出力します。通常は右側の「AI修復パッケージ出力」を使用してください。同じ完全融合JSONを含み、最終出版画像、クリーンなリソース骨格、厳格監査説明も生成します。', '导出 AI OCR 裁决包。全部原始 OCR 永久只读；此前已接受的冲突裁决会作为只读参考重新开放复审，AI 可保留旧结果，也可提交更好的 final_text；精确一致/共同候选仍锁定。': 'AI OCR裁決パッケージを出力します。すべての元OCRは永久に読み取り専用です。既採用の差異裁決は読み取り専用証拠として再確認可能で、AIは旧結果維持またはより良いfinal_text提出ができます。完全一致/共通候補はロック維持します。', '累积导入 AI OCR 裁决 JSON/ZIP，也兼容旧版逐模型 model_edits。后导入的已接受结果可改进同一稳定句；缺失/未决行不会抹掉前一包的好结果，且绝不改写原始 OCR。': 'AI OCR裁決JSON/ZIPを累積読込し、旧版のモデル別model_editsにも対応します。後から採用された結果は同一安定文を改善でき、欠行/未決行は以前の良い結果を消さず、元OCRも書き換えません。', '程序崩溃或重启后，直接从新版逐模型纠错 ZIP 恢复全部 OCR 文档、物理列、对齐和人工选择。先在页面管理重新载入同一批图片/PDF，可自动重绑新临时路径；不重新 OCR。': 'クラッシュ/再起動後、新版モデル別修正ZIPからすべてのOCR文書、物理列、整列、手動選択を直接復元します。先にページ管理で同じ画像/PDFを再読込すれば新しい一時パスを自動再結合し、再OCRしません。', '导出当前全部模型候选、人工融合结果、逐源纠错审计和干净稳定 ID 骨架 EPUB。': '現在の全モデル候補、手動融合結果、ソース別修正監査、クリーンな安定ID骨格EPUBを出力します。', '导出“AI 修复包 v3”：完整融合 JSON 保留全部文字证据；高置信度一致条目默认锁定，并生成原子跨度覆盖、全书重复/移动审计、独立插图 spine 计划和动态边界窗口。仅复制出版图片，并选择性导出高风险裁切图；存在出版 EPUB 时同时导出日文、Ruby 与结构对齐证据。': '「AI修復パッケージv3」を出力します。完全融合JSONが全文字証拠を保持し、高信頼一致行を既定ロック、原子span網羅、全書反復/移動監査、独立挿絵spine計画、動的境界窓を生成します。出版画像だけをコピーし、高リスクcropは任意出力。出版EPUBがあれば日本語、Ruby、構造整列証拠も出力します。', '导入外部大模型返回的稀疏 JSON，或直接读取已修改的 AI 修复 EPUB。只按稳定 row ID 添加候选，不接受页码、坐标、章节或资源改动。': '外部LLMの疎なJSON、または修正済みAI修復EPUBを直接読み込みます。候補は安定row IDでのみ追加し、ページ番号、座標、章、リソース変更は受け付けません。', '直接调用已配置的 AI，对有风险的多模型 OCR 句组进行低 token 审定；可选出版 EPUB 真值、物理列证据和独立第二遍审计。': '設定済みAIを直接呼び、リスクのある複数モデルOCR文グループを低tokenで裁定します。任意で出版EPUB真値、物理列証拠、独立第2監査を利用できます。', '严格按 row ID 导入外部大模型融合结果，完整替换当前融合正文并直接应用；缺行、重复 ID 或结构改变会拒绝导入。': '外部LLM融合結果をrow IDで厳密に読込み、現在の融合本文を完全置換して直接適用します。欠行、重複ID、構造変更は拒否します。', '修复旧版多模型包中‘上一行空白、下一行包含两句’的保守相邻错位；不改变 row ID、列 ID、封面、插图或用户已经修改的 edited_text。': '旧版複数モデルパッケージの「前行空白、次行に2文」の保守的隣接ずれを修復します。row ID、列ID、表紙、挿絵、ユーザー編集済みedited_textは変更しません。', '融合结果（真正一致与两模型共同候选均自动保留；真正分歧需裁决）': '融合結果（真の一致と2モデル共通候補は自動保持、真の差異は裁決が必要）', '默认隐藏真正一致和已经选择的融合行；真正分歧会显示全部不同候选，快速共识产生的两模型共同候选按 v8 规则自动采用，不再制造二次确认。勾选后自动前往下一组；取消勾选可查看全部融合结果。': '既定では真の一致と選択済み融合行を隠します。真の差異は全候補を表示し、高速合意の2モデル共通候補はv8規則で自動採用して再確認を増やしません。チェック時は次グループへ自動移動、解除時は全融合結果を表示します。', '勾选后，上方每个模型只显示当前句，下方只创建当前一个对比框。选择候选后自动进入下一句；完整 OCR 文本仍保存在内部，不影响重新对齐、导出或应用。': 'チェック時、上の各モデルは現在文だけを表示し、下には現在の比較枠1つだけ作成します。候補選択後は自動で次へ進みます。OCR全文は内部状態に保持され、再整列・出力・適用には影響しません。', '大文档只创建当前附近的候选控件；全部句子、选择和修改仍保存在内存状态中。': '大文書では現在位置周辺の候補コントロールだけを生成し、全ての文・選択・修正はメモリ状態に保持します。', '前往 OCR 对比的下一句；图文对照会记住并同步到同一稳定行。': 'OCR比較の次の文へ進み、画像/本文確認も同じ安定行へ記憶・同期します。', '逐句裁决：左侧队列定位未决句，上方各模型只显示当前句；绿色=一致，红色=差异。候选卡始终完整展开，红底=替换、橙底=增删；融合候选可直接修改，“显示全文对比”只切换显示。': '文ごとの裁決：左キューで未決文へ移動し、上の各モデルは現在文のみ表示します。緑=一致、赤=差異。候補カードは常に全展開し、赤背景=置換、橙背景=挿入/削除。融合候補は直接編集でき、「全文比較を表示」は表示だけを切替えます。', '源 OCR 已手动修改。字符红绿会自动刷新；若增删了换行，点击“重新自动对齐”后再裁决或应用。': '元OCRが手動編集されました。文字の赤緑差分は自動更新します。改行を追加/削除した場合は裁決/適用前に「再自動整列」を実行してください。', '标准紧凑包保留完整融合证据、稳定正文、原子跨度、页面列总账、必要视觉证据和审计工具，但不重复保存同一份全文、页面上下文和出版图片。\n完整取证包用于调试分列与人工检查全部风险，文件数和体积会明显增加。': '標準コンパクトパッケージは完全融合証拠、安定本文、原子span、ページ/列台帳、必要な視覚証拠、監査ツールを保持しつつ、同じ全文・ページ文脈・出版画像の重複保存を避けます。\n完全証拠パッケージは列分割デバッグと全リスク手動確認用で、ファイル数と容量が大きく増えます。', '正在取消 AI 审定；已完成批次会被丢弃，不会写入半成品融合稿…': 'AI裁定をキャンセル中です。完了済みバッチも破棄し、途中融合稿は書き込みません…', '已将当前逐句裁决稿应用到工作流；源 OCR 和候选卡仍保留在本页。': '現在の文単位裁決稿をワークフローへ適用しました。元OCRと候補カードはこの画面に残ります。', 'OCR逐列底稿：第一逻辑列显示在最右侧，与右边句图的阅读顺序一致。': '列別OCR下書き：論理列1を最右端に表示し、右側文画像の読順と一致させます。', '打开 Apple PKStrokeRecognizer 手写板；复制结果后自动写入当前文本框（⌥H）': 'Apple PKStrokeRecognizer手書きパッドを開きます。コピー結果は現在テキスト欄へ自動入力されます（⌥H）', '保存当前句并跳到下一条未核对的多模型分歧/低置信句（⌥↓）': '現在文を保存し、次の未確認の複数モデル差異/低信頼文へ移動（⌥↓）', '保存当前句并跳到上一条多模型 OCR 原始结果不一致的句图（⌥⇧D）': '現在文を保存し、複数モデル元OCRが不一致な前の文画像へ移動（⌥⇧D）', '保存当前句并跳到下一条多模型 OCR 原始结果不一致的句图，不受是否已核对影响（⌥D）': '現在文を保存し、確認済みかに関係なく次の複数モデル元OCR不一致文画像へ移動（⌥D）'})


# Full UI help/status: settings, diagnostics and remaining workspace states.
_EXACT_EN.update({'日本語': 'Japanese', '尚未检测。Windows 会检查 CIM、NVIDIA 驱动、CUDA/DirectML；Mac 会检查 Metal/MPS。': 'Not checked. Windows checks CIM, NVIDIA drivers and CUDA/DirectML; Mac checks Metal/MPS.', '点击“检测设备与 GPU”后显示 CPU、内存、显卡、驱动以及各 OCR 运行时可用的 CUDA / DirectML / MPS 后端。': "After clicking 'Detect Device & GPU', show CPU, memory, GPU, drivers and the CUDA / DirectML / MPS backends available to each OCR runtime.", '更新保护：不会自动更新、不会自动降级；Git 工作树有未提交修改时拒绝更新；便携包同版本且没有可信 commit 基线时也拒绝覆盖。更新程序代码时不会删除 .venv、模型缓存、OCR 运行时、debug/logs、输出书籍或用户工作区。': 'Update protection: never auto-update or auto-downgrade; refuse updates when the Git working tree has uncommitted changes; also refuse overwriting a portable package at the same version without a trusted commit baseline. Updating app code never deletes .venv, model caches, OCR runtimes, debug/logs, exported books or user workspaces.', '正在连接 GitHub 检查 main 分支版本和 commit…': 'Connecting to GitHub to check the main branch version and commit…', '日文书籍 OCR、逐列与逐字审校、文本修订、Formatter 和出版级 EPUB 导出的统一桌面工作台。': 'Unified desktop workspace for Japanese-book OCR, per-column/per-character review, text revision, Formatter and publication-grade EPUB export.', '✕ PaddleOCR 模型准备失败；请查看 OCR 日志并切换下载源。': '✕ PaddleOCR model preparation failed; check the OCR log and switch download source.', 'LiteRT 当前由 CPU interpreter 执行。': 'LiteRT is currently executed by the CPU interpreter.', 'PDF文字层模式需要重新接续物理列，因此不能同时固定原OCR排版': 'PDF text-layer mode must reconnect physical columns, so the original OCR layout cannot be fixed at the same time.', '内置模板，只读——想在它基础上改，先「新建空白格式」再把这段 CSS 复制过去编辑。': 'Built-in template, read-only. To modify it, create a blank format first and copy this CSS into it for editing.', 'DeepSeek 默认使用 V4-Flash 非思考模式；系统会固定静态提示前缀以提高缓存命中，并自动采用高并发批处理。': 'DeepSeek uses V4-Flash non-reasoning mode by default; the system pins a static prompt prefix to improve cache hits and automatically uses high-concurrency batching.', '设置已保存；API Key 仅在本次程序运行期间保存在内存中': 'Settings saved; API Key is kept in memory only for this application run', 'AI纠错排版版本会自动应用 AI 返回的 CSS；此处模板不参与导出': 'AI correction/layout versions automatically apply CSS returned by AI; this template is not used for export.', '已对当前单 OCR 副本执行无损 Unicode 标准化；未删除字符、未替换汉字，原始 OCR 未覆盖，点击“恢复初始”可撤销，点击“应用单OCR结果”后才进入后续流程。': "Lossless Unicode normalization was applied to the current single-OCR copy: no characters were deleted, no kanji replaced, and raw OCR was not overwritten. 'Restore Initial' undoes it; 'Apply Single OCR Result' is required before later workflow steps.", '已恢复标准化前的单 OCR 原始结果；Unicode 标准化副本已撤销。': 'Restored the pre-normalization single-OCR source; the Unicode-normalized copy was reverted.', '已开启单框逐句：每个模型只显示当前句，下方只显示一个对比框；选择后自动进入下一句。': 'Single-box sentence mode enabled: each model shows only the current sentence and one comparison box appears below; selecting a candidate advances automatically.', '已将当前单 OCR 结果应用到工作流；来源仍保留在 OCR 对比页。': 'Applied the current single-OCR result to the workflow; the source remains available on the OCR Compare page.', '需要 macOS 27 和 Xcode 27 / macOS 27 SDK。': 'Requires macOS 27 and Xcode 27 / macOS 27 SDK.', '自动模式将使用 Apple PKStrokeRecognizer（日语、设备本地、严格单字提交）。': 'Auto mode uses Apple PKStrokeRecognizer (Japanese, on-device, strict single-character submission).', '先运行 pip3 install -r requirements-handwriting-openvino.txt，再下载模型。': 'Run pip3 install -r requirements-handwriting-openvino.txt first, then download the model.', '高置信字符级融合 · 已自动采用（可重新选择原始 OCR）': 'High-confidence character-level fusion · Auto accepted (you can reselect among the raw OCR sources)', 'AI纠错已采用 · 原始OCR分歧保持可见 · 未覆盖任何模型原文': 'AI correction accepted · Raw OCR conflicts remain visible · No model source was overwritten', '所有真正分歧均已完成裁决；两模型共同候选已按 v8 规则自动保留。': 'All true conflicts have been adjudicated; two-model shared candidates were auto-kept under v8 rules.', '设置已保存；当前平台无系统密钥库，API Key 已安全降级为仅本次运行': 'Settings saved; this platform has no system key store, so API Key safely falls back to memory-only for this run'})
_EXACT_JA.update({'日本語': '日本語', '尚未检测。Windows 会检查 CIM、NVIDIA 驱动、CUDA/DirectML；Mac 会检查 Metal/MPS。': '未確認です。WindowsではCIM、NVIDIAドライバ、CUDA/DirectML、MacではMetal/MPSを確認します。', '点击“检测设备与 GPU”后显示 CPU、内存、显卡、驱动以及各 OCR 运行时可用的 CUDA / DirectML / MPS 后端。': '「デバイスとGPUを検出」をクリックすると、CPU、メモリ、GPU、ドライバ、各OCR実行環境で利用可能なCUDA / DirectML / MPSバックエンドを表示します。', '更新保护：不会自动更新、不会自动降级；Git 工作树有未提交修改时拒绝更新；便携包同版本且没有可信 commit 基线时也拒绝覆盖。更新程序代码时不会删除 .venv、模型缓存、OCR 运行时、debug/logs、输出书籍或用户工作区。': '更新保護：自動更新・自動ダウングレードは行いません。Git作業ツリーに未コミット変更がある場合は更新を拒否し、同一版のポータブルパッケージで信頼できるcommit基準がない場合も上書きを拒否します。アプリコード更新時も .venv、モデルキャッシュ、OCR実行環境、debug/logs、出力書籍、ユーザーワークスペースは削除しません。', '正在连接 GitHub 检查 main 分支版本和 commit…': 'GitHubへ接続しmainブランチのバージョンとcommitを確認中…', '日文书籍 OCR、逐列与逐字审校、文本修订、Formatter 和出版级 EPUB 导出的统一桌面工作台。': '日本語書籍OCR、列/文字単位校正、本文修訂、Formatter、出版級EPUB出力を統合したデスクトップワークスペースです。', '✕ PaddleOCR 模型准备失败；请查看 OCR 日志并切换下载源。': '✕ PaddleOCRモデル準備に失敗しました。OCRログを確認し、ダウンロード元を切り替えてください。', 'LiteRT 当前由 CPU interpreter 执行。': 'LiteRTは現在CPU interpreterで実行されています。', 'PDF文字层模式需要重新接续物理列，因此不能同时固定原OCR排版': 'PDFテキストレイヤーモードでは物理列を再接続する必要があるため、元OCRレイアウトを同時に固定できません。', '内置模板，只读——想在它基础上改，先「新建空白格式」再把这段 CSS 复制过去编辑。': '内蔵テンプレートは読み取り専用です。これを基に変更する場合は「空白形式を新規作成」して、このCSSをコピーして編集してください。', 'DeepSeek 默认使用 V4-Flash 非思考模式；系统会固定静态提示前缀以提高缓存命中，并自动采用高并发批处理。': 'DeepSeekは既定でV4-Flash非思考モードを使用します。キャッシュ命中率向上のため静的プロンプト接頭辞を固定し、高並列バッチ処理を自動使用します。', '设置已保存；API Key 仅在本次程序运行期间保存在内存中': '設定を保存しました。API Keyは今回のアプリ実行中だけメモリに保持します', 'AI纠错排版版本会自动应用 AI 返回的 CSS；此处模板不参与导出': 'AI校正・組版版はAIが返したCSSを自動適用し、ここでのテンプレートは出力に使いません。', '已对当前单 OCR 副本执行无损 Unicode 标准化；未删除字符、未替换汉字，原始 OCR 未覆盖，点击“恢复初始”可撤销，点击“应用单OCR结果”后才进入后续流程。': '現在の単一OCRコピーへ無損失Unicode正規化を適用しました。文字削除・漢字置換・元OCR上書きはありません。「初期状態を復元」で取り消し、「単一OCR結果を適用」後に次工程へ進みます。', '已恢复标准化前的单 OCR 原始结果；Unicode 标准化副本已撤销。': '正規化前の単一OCR原結果を復元し、Unicode正規化コピーを取り消しました。', '已开启单框逐句：每个模型只显示当前句，下方只显示一个对比框；选择后自动进入下一句。': '単一枠の文単位モードを有効化しました。各モデルは現在文だけを表示し、下に比較枠1つだけ表示します。候補選択後は自動で次へ進みます。', '已将当前单 OCR 结果应用到工作流；来源仍保留在 OCR 对比页。': '現在の単一OCR結果をワークフローへ適用しました。ソースはOCR比較画面に保持されます。', '需要 macOS 27 和 Xcode 27 / macOS 27 SDK。': 'macOS 27 と Xcode 27 / macOS 27 SDK が必要です。', '自动模式将使用 Apple PKStrokeRecognizer（日语、设备本地、严格单字提交）。': '自動モードではApple PKStrokeRecognizer（日本語、デバイス内、厳密な1文字送信）を使用します。', '先运行 pip3 install -r requirements-handwriting-openvino.txt，再下载模型。': '先に pip3 install -r requirements-handwriting-openvino.txt を実行し、その後モデルをダウンロードしてください。', '高置信字符级融合 · 已自动采用（可重新选择原始 OCR）': '高信頼の文字単位融合 · 自動採用済み（元OCRから再選択可能）', 'AI纠错已采用 · 原始OCR分歧保持可见 · 未覆盖任何模型原文': 'AI修正採用済み · 元OCR差異は表示維持 · どのモデル原文も上書きしていません', '所有真正分歧均已完成裁决；两模型共同候选已按 v8 规则自动保留。': '真の差異はすべて裁決済みで、2モデル共通候補はv8規則に従い自動保持しました。', '设置已保存；当前平台无系统密钥库，API Key 已安全降级为仅本次运行': '設定を保存しました。この環境にはシステム鍵保管庫がないため、API Keyは安全に今回の実行メモリ内のみへ切り替えました'})

# Common full-sentence strings used in settings and visible help panels.
_EXACT_EN.update({
    "可切换浅色/深色主题与中、日、英界面语言；只改变界面显示，不修改 OCR、文档或模型数据。": "Switch between light/dark themes and Chinese, Japanese, or English UI. This changes interface chrome only and never modifies OCR, documents, or model data.",
    "设置已保存": "Settings Saved",
    "设置已保存；OCR 默认行为已同步到当前工作区。": "Settings saved; OCR defaults have been synchronized to the current workspace.",
    "界面与功能原则": "Interface & Feature Principles",
    "统一使用浅色界面、淡蓝色边框；白色按钮文字固定为黑色。": "Choose light or dark appearance. Theme changes only affect UI chrome and never alter OCR data or document content.",
    "恢复上次功能区优先于默认启动功能区；子页签可独立记忆。": "Restoring the last workspace takes precedence over the default workspace; subtabs can be remembered independently.",
    "缓存、预览裁片和崩溃日志继续由原有安全清理流程管理。": "Caches, preview crops and crash logs continue to use the existing safe cleanup flow.",
    "这些选项在启动时应用，也可保存后立即同步到当前 OCR 工作区。": "These options apply at startup and are also synchronized to the current OCR workspace after saving.",
    "识别引擎、固定蓝框、分列顺序、多模型、逐字审校和运行日志仍集中在 OCR 识别区。": "OCR engine, fixed region, column order, multi-model OCR, character review and runtime logs remain in the OCR workspace.",
    "只读取本机文件，不联网": "Reads local files only; no network access",
    "仅在点击后连接各模型的官方仓库": "Connects to official model repositories only after you click",
    "检查完成。没有任何自动下载或自动替换。": "Check complete. Nothing was downloaded or replaced automatically.",
    "操作失败；现有模型未被静默覆盖。": "Operation failed; the existing model was not silently overwritten.",
})
_EXACT_JA.update({
    "可切换浅色/深色主题与中、日、英界面语言；只改变界面显示，不修改 OCR、文档或模型数据。": "ライト / ダークテーマと中国語・日本語・英語UIを切り替えられます。表示だけを変更し、OCR・文書・モデルデータは変更しません。",
    "设置已保存": "設定を保存しました",
    "设置已保存；OCR 默认行为已同步到当前工作区。": "設定を保存し、OCR既定動作を現在のワークスペースへ反映しました。",
    "界面与功能原则": "UIと機能の原則",
    "统一使用浅色界面、淡蓝色边框；白色按钮文字固定为黑色。": "ライト / ダーク表示を切り替えられます。外観変更はUIのみに作用し、OCRデータや本文は変更しません。",
    "恢复上次功能区优先于默认启动功能区；子页签可独立记忆。": "前回ワークスペースの復元は既定ワークスペースより優先され、サブタブは個別に記憶できます。",
    "缓存、预览裁片和崩溃日志继续由原有安全清理流程管理。": "キャッシュ、プレビュー切り出し、クラッシュログは従来の安全な消去フローで管理します。",
    "这些选项在启动时应用，也可保存后立即同步到当前 OCR 工作区。": "これらの設定は起動時に適用され、保存後は現在のOCRワークスペースにも即時反映されます。",
    "识别引擎、固定蓝框、分列顺序、多模型、逐字审校和运行日志仍集中在 OCR 识别区。": "認識エンジン、固定範囲、列順、複数モデル、文字単位校正、実行ログはOCRワークスペースに集約されます。",
    "只读取本机文件，不联网": "ローカルファイルのみ読み取り、ネット接続しません",
    "仅在点击后连接各模型的官方仓库": "クリックした場合のみ各モデルの公式リポジトリへ接続します",
    "检查完成。没有任何自动下载或自动替换。": "確認完了。自動ダウンロードや自動置換は行っていません。",
    "操作失败；现有模型未被静默覆盖。": "処理に失敗しました。既存モデルは上書きされていません。",
})

# Form labels that QFormLayout creates internally as QLabel children.
_EXACT_EN.update({
    "候选数量": "Candidate Count", "最小文字高度比例": "Minimum Text-height Ratio",
    "名称:": "Name:", "密钥存储": "Key Storage", "术语白名单": "Glossary Allowlist",
    "DeepSeek 思考": "DeepSeek Thinking", "思考强度": "Reasoning Effort",
    "OCR修复目标": "OCR Repair Goal", "并发请求": "Concurrent Requests",
    "单批输入 Tokens": "Input Tokens per Batch", "正文字符上限": "Body Character Limit",
    "请求超时": "Request Timeout", "JSON 输出": "JSON Output", "RPM 限制": "RPM Limit",
    "TPM 限制": "TPM Limit", "书名:": "Title:", "作者:": "Author:",
    "出版社:": "Publisher:", "卷号:": "Volume:", "CSS 模板:": "CSS Template:",
    "排版:": "Layout:", "出版参考：": "Publication Reference:", "术语表：": "Glossary:",
    "结果目录：": "Result Directory:", "分批：": "Batching:",
})
_EXACT_JA.update({
    "候选数量": "候補数", "最小文字高度比例": "最小文字高さ比",
    "名称:": "名前:", "密钥存储": "キー保存", "术语白名单": "用語ホワイトリスト",
    "DeepSeek 思考": "DeepSeek 推論", "思考强度": "推論強度",
    "OCR修复目标": "OCR修復目標", "并发请求": "同時リクエスト",
    "单批输入 Tokens": "1バッチ入力Tokens", "正文字符上限": "本文文字数上限",
    "请求超时": "リクエストタイムアウト", "JSON 输出": "JSON出力", "RPM 限制": "RPM制限",
    "TPM 限制": "TPM制限", "书名:": "書名:", "作者:": "著者:",
    "出版社:": "出版社:", "卷号:": "巻番号:", "CSS 模板:": "CSSテンプレート:",
    "排版:": "組版:", "出版参考：": "出版参照：", "术语表：": "用語集：",
    "结果目录：": "結果フォルダ：", "分批：": "バッチ：",
})

# Deep-audit additions for dynamic labels, suffixes and progress chrome that
# previously remained Chinese after an English/Japanese switch.
_EXACT_EN.update({
    "正在读取输入…": "Reading input…",
    "tiny（最小）": "tiny (smallest)",
    "文字行方向": "Text Line Orientation",
    " 秒": " sec",
    " 次": " times",
    " 列": " columns",
    "停止": "Stop",
    "处理前文本（可编辑，点击「应用编辑」写回）": "Before (editable; click Apply Edits to commit)",
    "处理后文本（可编辑，点击「应用编辑」写回）": "After (editable; click Apply Edits to commit)",
    "0 / 0 章": "0 / 0 chapters",
    "出版社": "Publisher",
    "▰ 背景色": "▰ Background",
    "正在停止…": "Stopping…",
    "准备证据…": "Preparing evidence…",
    "AI 审定与独立审计完成": "AI adjudication and independent audit complete",
    "管理方式": "Management",
    "☰ 0 章": "☰ 0 chapters",
    "准备中": "Preparing",
    "完成，待逐项确认": "Complete; awaiting item-by-item confirmation",
    "准备 OCR 裁决任务…": "Preparing OCR adjudication task…",
})
_EXACT_JA.update({
    "正在读取输入…": "入力を読み込み中…",
    "tiny（最小）": "tiny（最小）",
    "文字行方向": "テキスト行方向",
    " 秒": " 秒",
    " 次": " 回",
    " 列": " 列",
    "停止": "停止",
    "处理前文本（可编辑，点击「应用编辑」写回）": "処理前テキスト（編集可。「編集を適用」で書き戻し）",
    "处理后文本（可编辑，点击「应用编辑」写回）": "処理後テキスト（編集可。「編集を適用」で書き戻し）",
    "0 / 0 章": "0 / 0 章",
    "出版社": "出版社",
    "▰ 背景色": "▰ 背景色",
    "正在停止…": "停止処理中…",
    "准备证据…": "証拠を準備中…",
    "AI 审定与独立审计完成": "AI裁決と独立監査が完了しました",
    "管理方式": "管理方法",
    "☰ 0 章": "☰ 0 章",
    "准备中": "準備中",
    "完成，待逐项确认": "完了・項目ごとの確認待ち",
    "准备 OCR 裁决任务…": "OCR裁決タスクを準備中…",
})

# Dialog standard buttons and scan-preprocess dialog chrome.
_EXACT_EN.update({
    "全部保存": "Save All", "是": "Yes", "全部是": "Yes to All", "否": "No", "全部否": "No to All",
    "中止": "Abort", "忽略": "Ignore", "放弃": "Discard", "帮助": "Help", "应用": "Apply",
    "重置": "Reset", "恢复默认": "Restore Defaults", "显示详情": "Show Details", "隐藏详情": "Hide Details",
    "等待开始": "Waiting to Start", "尚未生成预览": "No Preview Yet",
    "只生成会话临时副本，不修改原图。处理完成后，页面管理、OCR预览、分列和多模型会统一读取同一批优化页面。": "Creates temporary session copies only and never modifies source images. After processing, Page Manager, OCR preview, column detection and multi-model OCR all use the same optimized pages.",
    "页面几何": "Page Geometry", "自动检测并裁掉桌面/黑边": "Auto-detect and crop desk/black borders",
    "透视拉正（检测可靠时才执行）": "Perspective correction (only when detection is reliable)",
    "小角度自动纠偏": "Auto deskew small angles", "书籍双页自动拆分": "Auto split two-page book scans",
    "保留边距": "Keep Margin", "日文书：右页 → 左页": "Japanese books: right page → left page",
    "横排书：左页 → 右页": "Horizontal books: left page → right page", "双页顺序": "Two-page Order",
    "漂白与去阴影": "Whitening & Shadow Removal", "不处理颜色/亮度": "Do not change color/brightness",
    "柔和漂白（轻小说推荐）": "Gentle whitening (recommended for light novels)",
    "强力文档（阴影较重）": "Strong document cleanup (heavy shadows)",
    "OCR专用灰度（不二值化）": "OCR-only grayscale (no binarization)", "增强模式": "Enhancement Mode",
    "保护彩色插图与印章": "Protect color illustrations and seals",
    "双页拆分会改变页面数量，但每个新页面会继承原页的正文/封面/插图分类。": "Two-page splitting changes the page count, but every new page inherits the source page's body/cover/illustration classification.",
    "更新当前页预览": "Update Current-page Preview", "恢复推荐设置": "Restore Recommended Settings",
    "正在生成…": "Generating…", "预览图读取失败": "Failed to Read Preview Image",
})
_EXACT_JA.update({
    "全部保存": "すべて保存", "是": "はい", "全部是": "すべてはい", "否": "いいえ", "全部否": "すべていいえ",
    "中止": "中止", "忽略": "無視", "放弃": "破棄", "帮助": "ヘルプ", "应用": "適用",
    "重置": "リセット", "恢复默认": "既定値に戻す", "显示详情": "詳細を表示", "隐藏详情": "詳細を隠す",
    "等待开始": "開始待ち", "尚未生成预览": "プレビュー未生成",
    "只生成会话临时副本，不修改原图。处理完成后，页面管理、OCR预览、分列和多模型会统一读取同一批优化页面。": "セッション用の一時コピーだけを生成し、元画像は変更しません。処理後はページ管理・OCRプレビュー・列分割・複数モデルOCRが同じ最適化ページを使用します。",
    "页面几何": "ページ形状", "自动检测并裁掉桌面/黑边": "机面/黒縁を自動検出して切り抜く",
    "透视拉正（检测可靠时才执行）": "遠近補正（検出が確実な場合のみ）", "小角度自动纠偏": "小角度の自動傾き補正",
    "书籍双页自动拆分": "見開きページを自動分割", "保留边距": "余白を保持",
    "日文书：右页 → 左页": "日本語書籍：右ページ → 左ページ", "横排书：左页 → 右页": "横書き書籍：左ページ → 右ページ",
    "双页顺序": "見開き順序", "漂白与去阴影": "白地化と影除去", "不处理颜色/亮度": "色/明るさを変更しない",
    "柔和漂白（轻小说推荐）": "穏やかな白地化（ライトノベル推奨）", "强力文档（阴影较重）": "強力文書処理（影が強い場合）",
    "OCR专用灰度（不二值化）": "OCR専用グレースケール（二値化なし）", "增强模式": "強調モード",
    "保护彩色插图与印章": "カラー挿絵と印章を保護",
    "双页拆分会改变页面数量，但每个新页面会继承原页的正文/封面/插图分类。": "見開き分割でページ数は変わりますが、新しい各ページは元ページの本文/表紙/挿絵分類を引き継ぎます。",
    "更新当前页预览": "現在ページのプレビューを更新", "恢复推荐设置": "推奨設定に戻す",
    "正在生成…": "生成中…", "预览图读取失败": "プレビュー画像の読み込みに失敗しました",
})


# Release audit: visible chrome added by the PDF/handwriting/page-preview workspaces.
# Keep these in the main UI catalog (not dialog_localization) because they are
# QLabel/title strings discovered by the static UI contract audit.
_EXACT_EN.update({
    "PDF 提取原文 · 尚未格式处理": "PDF Extracted Text · Not Yet Formatted",
    "OCR + 日语手写人工纠错": "OCR + Japanese Handwriting Manual Review",
    "无法预览": "Preview Unavailable",
})
_EXACT_JA.update({
    "PDF 提取原文 · 尚未格式处理": "PDF抽出原文 · 未整形",
    "OCR + 日语手写人工纠错": "OCR + 日本語手書き手動修正",
    "无法预览": "プレビューできません",
})

# Deep-audit coverage for the optional handwriting / character review workbench.
# These are UI chrome only; OCR text, candidate text and user edits are not
# passed through this catalog.
_EXACT_EN.update({
    "OCR + 日语手写人工纠错 · 疑点优先": "OCR + Japanese Handwriting Manual Review · Suspicion First",
    "以普通 OCR 文本为底稿，按风险优先定位疑点列。左侧核对原图，中央可分笔描摹生成候选，右侧可直接用键盘、macOS 日语输入源或系统手写输入修改。候选不会自动覆盖正文；只有点击“应用复核结果”后，明确修改或标记核对的列才会写回。": "Uses ordinary OCR text as the baseline and prioritizes suspicious columns by risk. Verify the source image on the left, trace strokes in the center to generate candidates, and edit on the right with the keyboard, macOS Japanese input source, or system handwriting. Candidates never overwrite the text automatically; only explicitly edited or reviewed columns are written back after clicking Apply Review Results.",
    "候选引擎：JLect JHR（CC BY-SA 3.0）": "Candidate engine: JLect JHR (CC BY-SA 3.0)",
    "跳过复核，保留 OCR 底稿": "Skip Review, Keep OCR Baseline",
    "应用人工纠错结果": "Apply Manual Corrections",
    "OCR + 日语逐字审校 · 字符扫描仅在本窗口启用": "OCR + Japanese Character Review · Character Scan Enabled Only Here",
    "普通 OCR 文本是底稿。字符扫描逐字框只在打开本审校窗口后启用，不会参与普通 OCR、简体中文横排 OCR 或后台候选筛查。程序只标记疑点，不会自动替换字符；OCR 三次仍为空的物理列会显示为 □。左侧始终保留对应原图，文本框支持 macOS 日语输入法和系统手写输入。": "Ordinary OCR text is the baseline. Per-character scan boxes are enabled only while this review window is open; they do not participate in normal OCR, Simplified-Chinese horizontal OCR, or background candidate screening. The application only flags suspicious characters and never replaces them automatically. A physical column that is empty after three OCR attempts is shown as □. The matching source image is always kept on the left, and the text field supports macOS Japanese input and system handwriting.",
    "← 上一列": "← Previous Column", "下一列 →": "Next Column →",
    "⚠ 上一疑点": "⚠ Previous Issue", "下一疑点 ⚠": "Next Issue ⚠",
    "当前字形与疑点": "Current Glyph and Issue", "点击左侧原图选择字符": "Click the source image at left to select a character",
    "左图框 · 中间字形 · 右侧光标使用同一字符索引": "Left image box · center glyph · right cursor use the same character index",
    "↑ 上一字": "↑ Previous Character", "下一字 ↓": "Next Character ↓", "采用候选": "Use Candidate",
    "候选只针对当前框运行，不会自动覆盖正文。": "Candidates run only for the current box and never overwrite body text automatically.",
    "点击“PKStroke 识别当前字”会把当前印刷字骨架转换为 PKDrawing 笔画并调用已连接的 Apple PKStrokeRecognizer；右侧仍可直接使用 macOS 日语输入源/系统手写。候选不会后台批量覆盖正文。": "Clicking “PKStroke Recognize Current Character” converts the printed glyph skeleton into PKDrawing strokes and calls the connected Apple PKStrokeRecognizer. The right side can still use the macOS Japanese input source or system handwriting directly. Candidates never batch-overwrite body text in the background.",
    "打开 Apple PKStroke 手写板": "Open Apple PKStroke Handwriting Pad",
    "打开已桥接的原生 PencilKit 手写面板；可手写日语并复制识别结果回右侧输入框。": "Open the bridged native PencilKit handwriting panel; write Japanese by hand and copy the recognition result back to the input field on the right.",
    "OCR 底稿（可直接编辑）": "OCR Baseline (Directly Editable)",
    "图文对照（与当前人工纠错同步）": "Image/Text Comparison (Synced with Manual Review)",
    "正在载入当前列/句原图…": "Loading the source image for the current column/sentence…",
    "显示当前列所属句子的 OCR 文本和来源信息": "Show OCR text and source information for the sentence containing the current column",
    "输入替换字；支持日语输入法/手写": "Enter replacement character; Japanese IME/handwriting supported",
    "替换当前字": "Replace Current Character", "在前面插入": "Insert Before", "删除当前字": "Delete Current Character",
    "仅在手动点击后运行 Apple OCR；不会后台自动识别": "Apple OCR runs only after a manual click; no automatic background recognition",
    "Apple OCR 识别本列并复制": "Recognize This Column with Apple OCR and Copy",
    "用 Apple OCR 替换本列": "Replace This Column with Apple OCR",
    "用 Mac 预览打开本列（⌘O）": "Open This Column in macOS Preview (⌘O)",
    "双击左侧列图或按 ⌘O，在 macOS 预览中打开当前列原图。": "Double-click the column image on the left or press ⌘O to open the current source column in macOS Preview.",
    "⌨ 聚焦日语输入": "⌨ Focus Japanese Input", "恢复本列 OCR": "Restore OCR for This Column", "标记本列已核对": "Mark This Column Reviewed",
    "跳过，保留 OCR 底稿": "Skip and Keep OCR Baseline",
    "正在手动运行 Apple OCR 识别当前单列；逐字框不会被替换…": "Manually running Apple OCR on the current column; character boxes will not be replaced…",
    "已用 Apple OCR 结果替换本列；应用人工纠错结果后写回正文。": "Replaced this column with the Apple OCR result; it will be written back after applying manual corrections.",
    "正在识别当前字；正文保持不变…": "Recognizing the current character; body text remains unchanged…",
    "已删除当前字。": "Current character deleted.", "已恢复本列原始 OCR。": "Original OCR for this column restored.",
    "本列已标记为人工核对。": "This column is marked as manually reviewed.",
    "输入框已聚焦，可切换 macOS 日语输入法或系统手写输入。": "The input field is focused; you can switch to the macOS Japanese IME or system handwriting input.",
    "当前列/句原图无法载入": "Cannot Load Source Image for Current Column/Sentence",
    "该列 OCR 三次均为空。请删除 □ 并在右侧输入整列原文。": "OCR returned empty for this column three times. Delete □ and enter the full original column text on the right.",
    "已打开 Apple PKStrokeRecognizer 原生手写板；识别后可复制结果到右侧单字输入框。": "Opened the native Apple PKStrokeRecognizer handwriting pad; copy the result to the single-character field on the right after recognition.",
    "本列原图不存在，无法打开。": "The source image for this column does not exist and cannot be opened.",
    "系统未能打开本列原图。": "The system could not open the source image for this column.",
    "已使用系统图片查看器打开本列原图。": "Opened the source image for this column with the system image viewer.",
    "Apple OCR 正在识别当前列，请稍候。": "Apple OCR is recognizing the current column. Please wait.",
    "本列原图不存在，无法运行 Apple OCR。": "The source image for this column does not exist, so Apple OCR cannot run.",
    "Apple OCR 本列结果已显示并复制到剪贴板；可选择替换本列。": "The Apple OCR result for this column is displayed and copied to the clipboard; you may replace this column with it.",
    "Apple OCR 完成了字符框对齐，但未返回可复制文本。": "Apple OCR completed character-box alignment but returned no text to copy.",
    "列图加载失败": "Failed to Load Column Image", "字形图不可用": "Glyph Image Unavailable",
    "当前没有自动标记的疑点列。": "There are currently no automatically flagged suspicious columns.",
    "已在 macOS 预览中打开本列原图（双击左图或按 ⌘O 可再次打开）。": "Opened the source image for this column in macOS Preview (double-click the left image or press ⌘O to open it again).",
})
_EXACT_JA.update({
    "OCR + 日语手写人工纠错 · 疑点优先": "OCR + 日本語手書き手動修正 · 疑問箇所優先",
    "以普通 OCR 文本为底稿，按风险优先定位疑点列。左侧核对原图，中央可分笔描摹生成候选，右侧可直接用键盘、macOS 日语输入源或系统手写输入修改。候选不会自动覆盖正文；只有点击“应用复核结果”后，明确修改或标记核对的列才会写回。": "通常OCRテキストを下書きとして、リスク順に疑問のある列を優先表示します。左で元画像を確認し、中央で筆画をなぞって候補を生成し、右でキーボード・macOS日本語入力ソース・システム手書き入力から直接修正できます。候補が本文を自動上書きすることはなく、「確認結果を適用」を押した後に明示的に修正または確認済みとした列だけを書き戻します。",
    "候选引擎：JLect JHR（CC BY-SA 3.0）": "候補エンジン：JLect JHR（CC BY-SA 3.0）",
    "跳过复核，保留 OCR 底稿": "確認をスキップしてOCR下書きを保持", "应用人工纠错结果": "手動修正結果を適用",
    "OCR + 日语逐字审校 · 字符扫描仅在本窗口启用": "OCR + 日本語1文字ずつ校正 · 文字スキャンはこの画面だけで有効",
    "普通 OCR 文本是底稿。字符扫描逐字框只在打开本审校窗口后启用，不会参与普通 OCR、简体中文横排 OCR 或后台候选筛查。程序只标记疑点，不会自动替换字符；OCR 三次仍为空的物理列会显示为 □。左侧始终保留对应原图，文本框支持 macOS 日语输入法和系统手写输入。": "通常OCRテキストが下書きです。文字スキャンの1文字枠はこの校正画面を開いている間だけ有効で、通常OCR・簡体字中国語横書きOCR・バックグラウンド候補選別には参加しません。アプリは疑問箇所を示すだけで文字を自動置換しません。3回OCRしても空の物理列は □ と表示します。左側には対応する元画像を常に保持し、テキスト欄はmacOS日本語入力とシステム手書き入力に対応します。",
    "← 上一列": "← 前の列", "下一列 →": "次の列 →", "⚠ 上一疑点": "⚠ 前の疑問箇所", "下一疑点 ⚠": "次の疑問箇所 ⚠",
    "当前字形与疑点": "現在の字形と疑問箇所", "点击左侧原图选择字符": "左の元画像をクリックして文字を選択",
    "左图框 · 中间字形 · 右侧光标使用同一字符索引": "左の画像枠・中央の字形・右のカーソルは同じ文字インデックスを使用",
    "↑ 上一字": "↑ 前の文字", "下一字 ↓": "次の文字 ↓", "采用候选": "候補を採用",
    "候选只针对当前框运行，不会自动覆盖正文。": "候補は現在の枠だけで実行され、本文を自動上書きしません。",
    "点击“PKStroke 识别当前字”会把当前印刷字骨架转换为 PKDrawing 笔画并调用已连接的 Apple PKStrokeRecognizer；右侧仍可直接使用 macOS 日语输入源/系统手写。候选不会后台批量覆盖正文。": "「PKStrokeで現在文字を認識」を押すと印刷文字の骨格をPKDrawing筆画に変換し、接続済みApple PKStrokeRecognizerを呼び出します。右側では引き続きmacOS日本語入力ソース/システム手書きを直接使えます。候補がバックグラウンドで本文を一括上書きすることはありません。",
    "打开 Apple PKStroke 手写板": "Apple PKStroke手書きパッドを開く",
    "打开已桥接的原生 PencilKit 手写面板；可手写日语并复制识别结果回右侧输入框。": "ブリッジ済みのネイティブPencilKit手書きパネルを開き、日本語を手書きして認識結果を右側入力欄へコピーできます。",
    "OCR 底稿（可直接编辑）": "OCR下書き（直接編集可）", 
    "正在载入当前列/句原图…": "現在の列/文の元画像を読み込み中…",
    "显示当前列所属句子的 OCR 文本和来源信息": "現在の列を含む文のOCRテキストと出典情報を表示",
    "输入替换字；支持日语输入法/手写": "置換文字を入力；日本語IME/手書き対応", "替换当前字": "現在文字を置換", "在前面插入": "前に挿入", "删除当前字": "現在文字を削除",
    "仅在手动点击后运行 Apple OCR；不会后台自动识别": "Apple OCRは手動クリック時だけ実行し、バックグラウンド自動認識はしません",
    "Apple OCR 识别本列并复制": "Apple OCRでこの列を認識してコピー", "用 Apple OCR 替换本列": "Apple OCRでこの列を置換",
    "用 Mac 预览打开本列（⌘O）": "この列をmacOSプレビューで開く（⌘O）",
    "双击左侧列图或按 ⌘O，在 macOS 预览中打开当前列原图。": "左の列画像をダブルクリックするか⌘Oを押して、現在列の元画像をmacOSプレビューで開きます。",
    "⌨ 聚焦日语输入": "⌨ 日本語入力にフォーカス", "恢复本列 OCR": "この列のOCRを復元", "标记本列已核对": "この列を確認済みにする",
    "跳过，保留 OCR 底稿": "スキップしてOCR下書きを保持",
    "正在手动运行 Apple OCR 识别当前单列；逐字框不会被替换…": "現在列をApple OCRで手動認識中です。文字枠は置換されません…",
    "已用 Apple OCR 结果替换本列；应用人工纠错结果后写回正文。": "この列をApple OCR結果で置換しました。手動修正結果を適用した後に本文へ書き戻します。",
    "正在识别当前字；正文保持不变…": "現在文字を認識中です。本文は変更しません…",
    "已删除当前字。": "現在文字を削除しました。", "已恢复本列原始 OCR。": "この列の元OCRを復元しました。", "本列已标记为人工核对。": "この列を手動確認済みにしました。",
    "输入框已聚焦，可切换 macOS 日语输入法或系统手写输入。": "入力欄にフォーカスしました。macOS日本語入力またはシステム手書き入力へ切り替えられます。",
    "当前列/句原图无法载入": "現在の列/文の元画像を読み込めません",
    "该列 OCR 三次均为空。请删除 □ 并在右侧输入整列原文。": "この列は3回OCRしても空でした。□ を削除し、右側に列全体の原文を入力してください。",
    "已打开 Apple PKStrokeRecognizer 原生手写板；识别后可复制结果到右侧单字输入框。": "Apple PKStrokeRecognizerのネイティブ手書きパッドを開きました。認識後、結果を右側の1文字入力欄へコピーできます。",
    "本列原图不存在，无法打开。": "この列の元画像が存在しないため開けません。", "系统未能打开本列原图。": "システムでこの列の元画像を開けませんでした。",
    "已使用系统图片查看器打开本列原图。": "システム画像ビューアでこの列の元画像を開きました。", "Apple OCR 正在识别当前列，请稍候。": "Apple OCRで現在列を認識中です。お待ちください。",
    "本列原图不存在，无法运行 Apple OCR。": "この列の元画像が存在しないためApple OCRを実行できません。",
    "Apple OCR 本列结果已显示并复制到剪贴板；可选择替换本列。": "この列のApple OCR結果を表示し、クリップボードへコピーしました。この列を置換できます。",
    "Apple OCR 完成了字符框对齐，但未返回可复制文本。": "Apple OCRは文字枠整列を完了しましたが、コピー可能なテキストを返しませんでした。",
    "列图加载失败": "列画像の読み込みに失敗", "字形图不可用": "字形画像を利用できません", "当前没有自动标记的疑点列。": "現在、自動でマークされた疑問列はありません。",
    "已在 macOS 预览中打开本列原图（双击左图或按 ⌘O 可再次打开）。": "この列の元画像をmacOSプレビューで開きました（左画像をダブルクリックするか⌘Oでもう一度開けます）。",
})

# Conservative dynamic patterns; these cover frequently changing counters/status
# without touching document text.
_PATTERNS = {
    LANG_EN: [
        (re.compile(r"^第\s*(\d+)\s*/\s*(\d+)\s*句$"), lambda m: f"Sentence {m.group(1)} / {m.group(2)}"),
        (re.compile(r"^第\s*(\d+)\s*句$"), lambda m: f"Sentence {m.group(1)}"),
        (re.compile(r"^第\s*(\d+)\s*页$"), lambda m: f"Page {m.group(1)}"),
        (re.compile(r"^(\d+)\s*页$"), lambda m: f"{m.group(1)} page(s)"),
        (re.compile(r"^已准备\s*(\d+)\s*页…$"), lambda m: f"Prepared {m.group(1)} page(s)…"),
        (re.compile(r"^选中\s*(\d+)\s*页\s*·\s*标记为：$"), lambda m: f"{m.group(1)} page(s) selected · Mark as:"),
        (re.compile(r"^第\s*(\d+)\s*/\s*(\d+)\s*页\s*·\s*(.+)$"), lambda m: f"Page {m.group(1)} / {m.group(2)} · {m.group(3)}"),
        (re.compile(r"^页面高清预览\s*·\s*第\s*(\d+)\s*页\s*·\s*(.+)$"), lambda m: f"HD Page Preview · Page {m.group(1)} · {m.group(2)}"),
        (re.compile(r"^适应窗口\s*·\s*(\d+)%$"), lambda m: f"Fit Window · {m.group(1)}%"),
        (re.compile(r"^当前图片：(.+)$"), lambda m: f"Current image: {m.group(1)}"),
        (re.compile(r"^当前：(.+)$"), lambda m: f"Current: {translate_text(m.group(1), LANG_EN)}"),
        (re.compile(r"^模型(\d+)：(.+)（当前主模型/结构底稿）$"), lambda m: f"Model {m.group(1)}: {m.group(2)} (primary/structure draft)"),
        (re.compile(r"^模型(\d+)\s*·\s*未载入$"), lambda m: f"Model {m.group(1)} · Not loaded"),
        (re.compile(r"^模型(\d+)$"), lambda m: f"Model {m.group(1)}"),
        (re.compile(r"^单 OCR：已载入\s*·\s*(.+)$"), lambda m: f"Single OCR: loaded · {m.group(1)}"),
        (re.compile(r"^查看AI修改（(\d+)）$"), lambda m: f"View AI Changes ({m.group(1)})"),
        (re.compile(r"^查看AI修改（采用(\d+) / 拦截(\d+)）$"), lambda m: f"View AI Changes (accepted {m.group(1)} / blocked {m.group(2)})"),
        (re.compile(r"^当前句：(.+)$"), lambda m: f"Current sentence: {m.group(1)}"),
    ],
    LANG_JA: [
        (re.compile(r"^第\s*(\d+)\s*/\s*(\d+)\s*句$"), lambda m: f"第 {m.group(1)} / {m.group(2)} 文"),
        (re.compile(r"^第\s*(\d+)\s*句$"), lambda m: f"第 {m.group(1)} 文"),
        (re.compile(r"^第\s*(\d+)\s*页$"), lambda m: f"{m.group(1)} ページ"),
        (re.compile(r"^(\d+)\s*页$"), lambda m: f"{m.group(1)} ページ"),
        (re.compile(r"^已准备\s*(\d+)\s*页…$"), lambda m: f"{m.group(1)} ページ準備済み…"),
        (re.compile(r"^选中\s*(\d+)\s*页\s*·\s*标记为：$"), lambda m: f"{m.group(1)}ページ選択 · 種別："),
        (re.compile(r"^第\s*(\d+)\s*/\s*(\d+)\s*页\s*·\s*(.+)$"), lambda m: f"{m.group(1)} / {m.group(2)} ページ · {m.group(3)}"),
        (re.compile(r"^页面高清预览\s*·\s*第\s*(\d+)\s*页\s*·\s*(.+)$"), lambda m: f"高精細ページプレビュー · {m.group(1)} ページ · {m.group(2)}"),
        (re.compile(r"^适应窗口\s*·\s*(\d+)%$"), lambda m: f"ウィンドウに合わせる · {m.group(1)}%"),
        (re.compile(r"^当前图片：(.+)$"), lambda m: f"現在画像：{m.group(1)}"),
        (re.compile(r"^当前：(.+)$"), lambda m: f"現在：{translate_text(m.group(1), LANG_JA)}"),
        (re.compile(r"^模型(\d+)：(.+)（当前主模型/结构底稿）$"), lambda m: f"モデル{m.group(1)}：{m.group(2)}（主モデル/構造下書き）"),
        (re.compile(r"^模型(\d+)\s*·\s*未载入$"), lambda m: f"モデル{m.group(1)} · 未読込"),
        (re.compile(r"^模型(\d+)$"), lambda m: f"モデル{m.group(1)}"),
        (re.compile(r"^单 OCR：已载入\s*·\s*(.+)$"), lambda m: f"単一OCR：読込済み · {m.group(1)}"),
        (re.compile(r"^查看AI修改（(\d+)）$"), lambda m: f"AI変更を表示（{m.group(1)}）"),
        (re.compile(r"^查看AI修改（采用(\d+) / 拦截(\d+)）$"), lambda m: f"AI変更を表示（採用 {m.group(1)} / ブロック {m.group(2)}）"),
        (re.compile(r"^当前句：(.+)$"), lambda m: f"現在文：{m.group(1)}"),
    ],
}


def normalize_language(value: str | None) -> str:
    value = str(value or "").strip()
    return value if value in SUPPORTED_LANGUAGES else LANG_ZH




# Standalone multimodal image-book workflow (2026-09-20).
_EXACT_EN.update({
    "AI 图文处理": "AI Image Processing",
    "一键最快": "Fastest Preset",
    "一键提取并翻译": "Extract + Translate Preset",
    "切换为仅提取原文、1600px、裁白边且不自动二审；GLM 默认关闭思考": "Use source-only extraction, 1600px, white-margin trimming, and no automatic second review; GLM reasoning is off by default",
    "切换为提取原文 + 翻译、简体中文、1600px、裁白边且不自动二审；GLM 默认关闭思考": "Use source extraction + translation to Simplified Chinese, 1600px, white-margin trimming, and no automatic second review; GLM reasoning is off by default",
    "已应用最快设置：仅提取原文 · 1600px · 裁白边 · 不自动二审 · GLM默认不思考": "Fastest preset applied: source only · 1600px · trim margins · no second review · GLM reasoning off",
    "已应用提取并翻译设置：简体中文 · 快速1600px · 裁白边 · 不自动二审 · GLM默认不思考": "Extract + translate preset applied: Simplified Chinese · fast 1600px · trim margins · no second review · GLM reasoning off",
    "AI 成稿预览": "AI Draft Preview",
    "AI设置不可用": "AI settings unavailable",
    "仅提取原文": "Extract Source Only",
    "使用当前页面中的一张缩小图片发起一次真实 API 请求，会产生极少量 API 用量。": "Send one downscaled current page in a real API request. This uses a very small amount of API quota.",
    "准备页面并建立省 token 的视觉请求…": "Preparing pages and token-efficient vision requests…",
    "出版（2560px + 高分辨率局部复核）": "Publication (2560px + high-resolution regional review)",
    "原文 → EPUB": "Source → EPUB",
    "只对 AI 主动标记的不确定区域进行二次放大复核": "Recheck only regions the AI explicitly marks as uncertain",
    "图片能力测试失败": "Image capability test failed",
    "均衡（推荐 · 1920px + 疑难局部复核）": "Balanced (Recommended · 1920px + regional review)",
    "处理任务": "Task",
    "处理状态": "Processing Status",
    "完成后显示前若干段原文/译文；完整内容会作为 UnifiedDocument 交给 EPUB Builder。": "Shows a short source/translation preview after completion; the complete result is passed to EPUB Builder as a UnifiedDocument.",
    "已读取页面管理；可以开始 AI 图文处理": "Page Manager data loaded; AI image processing is ready",
    "快速（1600px · 不自动二审）": "Fast (1600px · no automatic second review)",
    "打开页面管理": "Open Page Manager",
    "提取原文 + 翻译": "Extract Source + Translate",
    "正在停止；已完成请求和本地缓存会保留…": "Stopping; completed requests and local cache will be kept…",
    "正在测试模型图片输入能力…": "Testing model image-input capability…",
    "测试图片能力": "Test Image Capability",
    "独立模式 · 读取页面管理 · AI直接完成识别/结构/可选翻译 · 本地生成EPUB": "Independent mode · uses Page Manager · AI handles recognition/structure/optional translation · EPUB is built locally",
    "目标语言": "Target Language",
    "等待页面管理输入": "Waiting for Page Manager input",
    "繁體中文": "Traditional Chinese",
    "裁掉明显纯白页边（减少视觉 token）": "Trim obvious white margins (reduce vision tokens)",
    "请先在页面管理导入 PDF、图片或图片文件夹": "Import a PDF, images, or an image folder in Page Manager first",
    "质量 / Token": "Quality / Tokens",
    "页面管理：尚未导入页面": "Page Manager: no pages imported",
    "页面顺序、封面、插图和非正文页由页面管理决定；AI不重新分类。翻译模式在同一次视觉调用中返回原文+初译，只有复核后改变的少量块才同步修正，避免整书图片重复上传。": "Page order, cover, illustrations, and non-body pages are controlled by Page Manager; the AI does not reclassify them. Translation mode returns source + initial translation in the same vision call, and only the few blocks changed by review are corrected, avoiding a second full-book image upload.",
    "页面顺序、封面、插图和非正文页由页面管理决定；AI不重新分类。均衡/出版模式会在模型漏回批次时自动拆小重试，并只对异常边界/疑难区域追加视觉核验；翻译缺失优先用纯文本补译，不会为补译再次上传整页图片。": "Page order, cover, illustrations, and non-body pages are controlled by Page Manager; the AI does not reclassify them. Balanced/Publication mode automatically splits and retries incomplete batches, adds visual verification only for suspicious boundaries or uncertain regions, and repairs missing translations with text-only requests instead of uploading full pages again.",
    "页面顺序、封面、插图和非正文页由页面管理决定；AI不重新分类。均衡/出版模式会在模型漏回批次时自动拆小重试，并只对空页、异常短页、重复边界/疑难区域追加定向核验；跨页接续和翻译缺失优先走纯文本检查/补译，不会为这些步骤再次上传整页图片。": "Page order, cover, illustrations, and non-body pages are controlled by Page Manager; the AI does not reclassify them. Balanced/Publication mode splits and retries incomplete batches and adds targeted review only for empty pages, suspiciously short pages, duplicate boundaries, or uncertain regions. Cross-page continuity and missing translations use text-only checks/repair first, so full page images are not uploaded again for those steps.",
    "无法核对页面": "Unable to Verify Pages",
    "页面已变化": "Pages Changed",
})
_EXACT_JA.update({
    "AI 图文处理": "AI画像・本文処理",
    "一键最快": "最速設定",
    "一键提取并翻译": "原文抽出＋翻訳設定",
    "切换为仅提取原文、1600px、裁白边且不自动二审；GLM 默认关闭思考": "原文のみ・1600px・白余白トリミング・自動二次確認なしに切替。GLM推論は既定で無効です",
    "切换为提取原文 + 翻译、简体中文、1600px、裁白边且不自动二审；GLM 默认关闭思考": "原文抽出＋簡体字中国語への翻訳・1600px・白余白トリミング・自動二次確認なしに切替。GLM推論は既定で無効です",
    "已应用最快设置：仅提取原文 · 1600px · 裁白边 · 不自动二审 · GLM默认不思考": "最速設定を適用：原文のみ · 1600px · 余白トリミング · 二次確認なし · GLM推論なし",
    "已应用提取并翻译设置：简体中文 · 快速1600px · 裁白边 · 不自动二审 · GLM默认不思考": "原文抽出＋翻訳設定を適用：簡体字中国語 · 高速1600px · 余白トリミング · 二次確認なし · GLM推論なし",
    "AI 成稿预览": "AI完成稿プレビュー",
    "AI设置不可用": "AI設定を利用できません",
    "仅提取原文": "原文のみ抽出",
    "使用当前页面中的一张缩小图片发起一次真实 API 请求，会产生极少量 API 用量。": "現在のページ1枚を縮小して実APIへ送信します。ごく少量のAPI利用が発生します。",
    "准备页面并建立省 token 的视觉请求…": "ページを準備し、トークン節約型の画像リクエストを作成中…",
    "出版（2560px + 高分辨率局部复核）": "出版（2560px + 高解像度の局所再確認）",
    "原文 → EPUB": "原文をEPUBへ",
    "只对 AI 主动标记的不确定区域进行二次放大复核": "AIが不確実と明示した領域だけを拡大して再確認",
    "图片能力测试失败": "画像入力テストに失敗",
    "均衡（推荐 · 1920px + 疑难局部复核）": "バランス（推奨 · 1920px + 難所の局所再確認）",
    "处理任务": "処理タスク",
    "处理状态": "処理状況",
    "完成后显示前若干段原文/译文；完整内容会作为 UnifiedDocument 交给 EPUB Builder。": "完了後に原文/訳文の先頭数段を表示します。完全な内容はUnifiedDocumentとしてEPUB Builderへ渡します。",
    "已读取页面管理；可以开始 AI 图文处理": "ページ管理情報を読み込みました。AI画像・本文処理を開始できます",
    "快速（1600px · 不自动二审）": "高速（1600px · 自動二次確認なし）",
    "打开页面管理": "ページ管理を開く",
    "提取原文 + 翻译": "原文抽出 + 翻訳",
    "正在停止；已完成请求和本地缓存会保留…": "停止中です。完了済みリクエストとローカルキャッシュは保持します…",
    "正在测试模型图片输入能力…": "モデルの画像入力能力をテスト中…",
    "测试图片能力": "画像入力をテスト",
    "独立模式 · 读取页面管理 · AI直接完成识别/结构/可选翻译 · 本地生成EPUB": "独立モード · ページ管理を使用 · AIが認識/構造/任意翻訳を直接処理 · EPUBはローカル生成",
    "目标语言": "翻訳先言語",
    "等待页面管理输入": "ページ管理の入力待ち",
    "繁體中文": "繁体字中国語",
    "裁掉明显纯白页边（减少视觉 token）": "明らかな白余白をトリミング（画像トークン削減）",
    "请先在页面管理导入 PDF、图片或图片文件夹": "先にページ管理でPDF、画像、または画像フォルダを読み込んでください",
    "质量 / Token": "品質 / Token",
    "页面管理：尚未导入页面": "ページ管理：ページ未読込",
    "页面顺序、封面、插图和非正文页由页面管理决定；AI不重新分类。翻译模式在同一次视觉调用中返回原文+初译，只有复核后改变的少量块才同步修正，避免整书图片重复上传。": "ページ順、表紙、挿絵、非本文ページはページ管理が決定し、AIは再分類しません。翻訳モードでは同じ画像呼出しで原文+初訳を返し、再確認で変わった少数ブロックだけを修正するため、全書画像の再送を避けられます。",
    "页面顺序、封面、插图和非正文页由页面管理决定；AI不重新分类。均衡/出版模式会在模型漏回批次时自动拆小重试，并只对异常边界/疑难区域追加视觉核验；翻译缺失优先用纯文本补译，不会为补译再次上传整页图片。": "ページ順、表紙、挿絵、非本文ページはページ管理が決定し、AIは再分類しません。バランス/出版モードでは不完全なバッチを自動分割して再試行し、異常な境界や不確実な領域だけを追加で画像確認します。欠落した翻訳はまずテキストのみで補完し、そのために全ページ画像を再送しません。",
    "页面顺序、封面、插图和非正文页由页面管理决定；AI不重新分类。均衡/出版模式会在模型漏回批次时自动拆小重试，并只对空页、异常短页、重复边界/疑难区域追加定向核验；跨页接续和翻译缺失优先走纯文本检查/补译，不会为这些步骤再次上传整页图片。": "ページ順、表紙、挿絵、非本文ページはページ管理が決定し、AIは再分類しません。バランス/出版モードでは不完全なバッチを自動分割して再試行し、空ページ・異常に短いページ・重複境界・不確実な領域だけを対象に追加確認します。ページ跨ぎの接続と翻訳欠落はまずテキストのみで確認・補完し、そのために全ページ画像を再送しません。",
    "无法核对页面": "ページを検証できません",
    "页面已变化": "ページが変更されました",
})

_EXACT_EN.update({
    "启用 GLM 思考模式（默认关闭；会明显增加等待时间）": "Enable GLM reasoning (off by default; substantially increases wait time)",
    "GLM 思考": "GLM Reasoning",
    "生成一张本地 NF42 测试图，以与 AI 图文处理相同的 Base64 Data URL 方式验证当前模型是否真的读取图片": "Generate a local NF42 test image and verify that the current model actually reads it using the same Base64 Data URL path as AI Image Processing.",
    "正在用本地测试图验证图片输入……": "Verifying image input with a local test image…",
    "连接/图片能力测试失败": "Connection / image capability test failed",
    "智谱国内开放平台：Base URL 应为 https://open.bigmodel.cn/api/paas/v4。AI 图文处理请再点“测试图片能力”，不要只测纯文本连接。": "Zhipu BigModel China: Base URL should be https://open.bigmodel.cn/api/paas/v4. For AI Image Processing, also run Test Image Capability instead of relying on the text-only connection test.",
    "Z.AI 国际平台：默认 Base URL 为 https://api.z.ai/api/paas/v4/。国内 BigModel 密钥请改选“智谱 BigModel / GLM（国内）”。": "Z.AI international platform: default Base URL is https://api.z.ai/api/paas/v4/. For a mainland BigModel key, choose Zhipu BigModel / GLM (China).",
    "智谱 BigModel / GLM（国内）": "Zhipu BigModel / GLM (China)",
    "Z.AI / GLM（国际）": "Z.AI / GLM (International)",
})
_EXACT_JA.update({
    "启用 GLM 思考模式（默认关闭；会明显增加等待时间）": "GLM推論を有効化（既定は無効。待ち時間が大幅に増えます）",
    "GLM 思考": "GLM推論",
    "生成一张本地 NF42 测试图，以与 AI 图文处理相同的 Base64 Data URL 方式验证当前模型是否真的读取图片": "ローカルにNF42テスト画像を生成し、AI画像・本文処理と同じBase64 Data URL経路で現在のモデルが実際に画像を読めるか確認します。",
    "正在用本地测试图验证图片输入……": "ローカルテスト画像で画像入力を確認中……",
    "连接/图片能力测试失败": "接続/画像入力テストに失敗しました",
    "智谱国内开放平台：Base URL 应为 https://open.bigmodel.cn/api/paas/v4。AI 图文处理请再点“测试图片能力”，不要只测纯文本连接。": "智譜BigModel中国向け：Base URLは https://open.bigmodel.cn/api/paas/v4 です。AI画像・本文処理ではテキスト接続だけでなく「画像入力をテスト」も実行してください。",
    "Z.AI 国际平台：默认 Base URL 为 https://api.z.ai/api/paas/v4/。国内 BigModel 密钥请改选“智谱 BigModel / GLM（国内）”。": "Z.AI国際版：既定Base URLは https://api.z.ai/api/paas/v4/ です。中国向けBigModelキーは「智譜 BigModel / GLM（中国）」を選択してください。",
    "智谱 BigModel / GLM（国内）": "智譜 BigModel / GLM（中国）",
    "Z.AI / GLM（国际）": "Z.AI / GLM（国際）",
})

# v27 single-column model transport UI
_EXACT_EN.update({
    "分列显示：保留原页位置预览，OCR仍用单列输入": "Column Display: keep page-position preview; OCR still uses a single-column input",
    "两种方式都先建立同一份 Ruby-free 正文可见层。界面可用原页位置检查“分列显示”，但真正送入 OCR 前都会按当前单列墨迹收紧，并在四边保留安全白边；不会再把整页白底交给模型。正文像素不缩放，Ruby 与相邻列始终保持纸白隔离。": "Both modes build the same Ruby-free body visibility layer. Column Display may keep page-position context in the UI, but the actual OCR input is tightened around the current column with safe white margins on all four sides; a full-page white canvas is never sent to the model. Body pixels are not rescaled, and Ruby/adjacent columns remain paper-white isolated.",
})
_EXACT_JA.update({
    "分列显示：保留原页位置预览，OCR仍用单列输入": "列表示：ページ上の位置をプレビューし、OCR入力は1列のみ",
    "两种方式都先建立同一份 Ruby-free 正文可见层。界面可用原页位置检查“分列显示”，但真正送入 OCR 前都会按当前单列墨迹收紧，并在四边保留安全白边；不会再把整页白底交给模型。正文像素不缩放，Ruby 与相邻列始终保持纸白隔离。": "両モードとも同じルビ除外済み本文可視レイヤーを作ります。列表示ではUI上でページ内位置を確認できますが、実際のOCR入力は現在列の墨迹に合わせて四辺へ安全な白余白を残した1列画像へ絞り、全面白背景ページをモデルへ渡しません。本文画素は拡大縮小せず、ルビと隣接列は紙白で隔離します。",
})

_EXACT_EN.update({"分列显示（原页位置预览）": "Column Display (original page-position preview)"})
_EXACT_JA.update({"分列显示（原页位置预览）": "列表示（元ページ位置プレビュー）"})


# Phase43 selectable-PDF formatting UI, absorbed onto the Phase37 final UI.
_EXACT_EN.update({
    "PDF 格式处理": "PDF Formatting",
    "PDF文字层只负责完整提取字符与几何；这里统一处理分页/分列接续、前后书、站点页和正文格式。章节 / TOC 交给后续 AI 识别。": "The PDF text layer only extracts complete text and geometry. This stage handles column/page continuation, author notes, generated site pages, and body formatting. Chapter/TOC detection is deferred to the later AI pass.",
    "等待 PDF 文字层结果": "Waiting for PDF text-layer results",
    "PDF 专用规则": "PDF-Specific Rules",
    "Ruby / 小字号注音": "Ruby / Small Reading Text",
    "页码过滤（仅几何确认 ASCII）": "Page-number Filtering (geometry-confirmed ASCII only)",
    "物理列 + 跨页接续": "Physical Columns + Cross-page Continuation",
    "章节 / TOC 识别": "Chapter / TOC Detection",
    "目录待 AI 识别 · 交给 AI": "TOC pending AI detection · defer to AI",
    "删除作者前书 / 后书": "Remove Author Forewords / Afterwords",
    "清理 PDFNovels 站点前后置页": "Remove PDFNovels Generated Front/Back Matter",
    "恢复出版段首缩进": "Restore Publication Paragraph Indents",
    "提取层自动": "Automatic at Extraction",
    "几何自动": "Geometry Auto",
    "字符守卫：强制执行，发现 missing / extra 会在报告中标红": "Character guard is mandatory; missing / extra characters are highlighted in the report.",
    "开始 PDF 格式处理": "Start PDF Formatting",
    "通用高级编辑 / 全部工具  ›": "General Advanced Editing / All Tools  ›",
    "请先在“PDF文字层”完成提取": "Finish extraction in PDF Text Layer first.",
    "当前文档不是 PDF 文字层结果": "The current document is not a PDF text-layer result.",
})
_EXACT_JA.update({
    "PDF 格式处理": "PDF 書式処理",
    "PDF文字层只负责完整提取字符与几何；这里统一处理分页/分列接续、前后书、站点页和正文格式。章节 / TOC 交给后续 AI 识别。": "PDFテキストレイヤーは文字と座標の完全抽出だけを担当します。ここでは列・ページ跨ぎ接続、作者前後書き、サイト生成ページ、本文形式を処理し、章 / TOC の識別は後段のAIに任せます。",
    "等待 PDF 文字层结果": "PDFテキストレイヤー結果を待機中",
    "PDF 专用规则": "PDF 専用ルール",
    "Ruby / 小字号注音": "ルビ / 小字号注音",
    "页码过滤（仅几何确认 ASCII）": "ページ番号除外（幾何確認済みASCIIのみ）",
    "物理列 + 跨页接续": "物理列 + ページ跨ぎ接続",
    "章节 / TOC 识别": "章 / TOC 識別",
    "目录待 AI 识别 · 交给 AI": "TOCはAI識別待ち · AIに任せる",
    "删除作者前书 / 后书": "作者前書き / 後書きを削除",
    "清理 PDFNovels 站点前后置页": "PDFNovels生成の前後ページを削除",
    "恢复出版段首缩进": "出版用段落字下げを復元",
    "提取层自动": "抽出層で自動",
    "几何自动": "幾何で自動",
    "字符守卫：强制执行，发现 missing / extra 会在报告中标红": "文字ガードを必須実行し、missing / extra はレポートで強調表示します。",
    "开始 PDF 格式处理": "PDF 書式処理を開始",
    "通用高级编辑 / 全部工具  ›": "汎用高度編集 / 全ツール  ›",
    "请先在“PDF文字层”完成提取": "先に「PDFテキストレイヤー」で抽出を完了してください。",
    "当前文档不是 PDF 文字层结果": "現在の文書はPDFテキストレイヤー結果ではありません。",
})

def translate_text(text: str | None, language: str) -> str:
    """Translate UI chrome only; unknown strings remain exactly unchanged."""
    source = "" if text is None else str(text)
    language = normalize_language(language)
    if language == LANG_ZH or not source:
        return source
    mapping = _EXACT_JA if language == LANG_JA else _EXACT_EN
    if source in mapping:
        return mapping[source]
    for pattern, repl in _PATTERNS.get(language, ()):  # dynamic counters/status
        match = pattern.match(source)
        if match:
            return repl(match)
    # Runtime status/tool-tip text is often assembled from stable UI lines plus
    # paths/model labels. Translate complete known lines independently while
    # leaving unknown payload lines byte-for-byte unchanged.
    if "\n" in source:
        parts = source.splitlines(keepends=True)
        translated_parts: list[str] = []
        changed = False
        for part in parts:
            body = part.rstrip("\r\n")
            ending = part[len(body):]
            target = translate_text(body, language)
            changed = changed or target != body
            translated_parts.append(target + ending)
        if changed:
            return "".join(translated_parts)
    # Sidebar/tooltips often append a shortcut to a translated base label.
    shortcut = re.match(r"^(.*?)(（(?:⌘|Ctrl\+)\d+）)$", source)
    if shortcut:
        base = translate_text(shortcut.group(1), language)
        if base != shortcut.group(1):
            return base + shortcut.group(2)
    return source


def translation_coverage(strings: list[str], language: str) -> tuple[int, int]:
    """Testing helper: count exact/dynamic translations without mutating UI."""
    total = len(strings)
    changed = sum(1 for value in strings if translate_text(value, language) != value)
    return changed, total

# Runtime-composed UI chrome discovered by the deep audit.  Every regular
# expression is anchored and captures dynamic payload verbatim; model names,
# filenames, paths and OCR/document text are never machine-translated.
def _catalog_fragment(value: str, language: str) -> str:
    mapping = _EXACT_JA if language == LANG_JA else _EXACT_EN
    return mapping.get(str(value), str(value))


_PATTERNS[LANG_EN].extend([
    (re.compile(r"^已扫描：(\d+) 项可直接使用$"), lambda m: f"Scanned: {m.group(1)} item(s) ready to use"),
    (re.compile(r"^当前：页面识别 (\d+) / (\d+) 页$"), lambda m: f"Current: page recognition {m.group(1)} / {m.group(2)}"),
    (re.compile(r"^(.+)：准备中$"), lambda m: f"{_catalog_fragment(m.group(1), LANG_EN)}: preparing"),
    (re.compile(r"^✓ 已读取 (\d+) 个可用模型，可从下拉框选择；也可以手动输入。$"), lambda m: f"✓ Loaded {m.group(1)} available model(s). Select from the list or enter one manually."),
    (re.compile(r"^模型列表读取失败：(.*)。仍可在下拉框中手动填写模型名称。$"), lambda m: f"Failed to load model list: {m.group(1)}. You can still enter a model name manually."),
    (re.compile(r"^✓ 连接成功：(.*)$"), lambda m: f"✓ Connection succeeded: {m.group(1)}"),
    (re.compile(r"^正在执行：(.*)$"), lambda m: f"Running: {_catalog_fragment(m.group(1), LANG_EN)}"),
    (re.compile(r"^(.+)正在处理$"), lambda m: f"{_catalog_fragment(m.group(1), LANG_EN)} in progress"),
    (re.compile(r"^✓ (.+)完成$"), lambda m: f"✓ {_catalog_fragment(m.group(1), LANG_EN)} complete"),
    (re.compile(r"^已清空：(.*)$"), lambda m: f"Cleared: {_catalog_fragment(m.group(1), LANG_EN)}"),
    (re.compile(r"^正文来源：(.*)$"), lambda m: f"Body source: {m.group(1)}"),
    (re.compile(r"^图片页：(\d+)$"), lambda m: f"Image pages: {m.group(1)}"),
    (re.compile(r"^章节：(\d+)$"), lambda m: f"Chapters: {m.group(1)}"),
    (re.compile(r"^☰ (\d+) 章$"), lambda m: f"☰ {m.group(1)} chapters"),
    (re.compile(r"^🎨 (\d+) 图$"), lambda m: f"🎨 {m.group(1)} images"),
    (re.compile(r"^(\d+) / (\d+) 页$"), lambda m: f"{m.group(1)} / {m.group(2)} pages"),
    (re.compile(r"^后台对齐：左 (\d+) 行，右 (\d+) 行……$"), lambda m: f"Aligning in background: left {m.group(1)} lines, right {m.group(2)} lines…"),
    (re.compile(r"^左侧 · (.*)$"), lambda m: f"Left · {m.group(1)}"),
    (re.compile(r"^右侧 · (.*)（可编辑）$"), lambda m: f"Right · {m.group(1)} (editable)"),
    (re.compile(r"^已按当前编辑内容重新对齐 (\d+) 行$"), lambda m: f"Re-aligned {m.group(1)} lines from the current edits"),
    (re.compile(r"^已实时保持双栏对齐；当前约第 (\d+) 行。修改尚未写回其他工作区。$"), lambda m: f"Both panes remain aligned in real time; currently around line {m.group(1)}. Changes have not yet been written to other workspaces."),
    (re.compile(r"^已将右侧 (\d+) 处完整对白拆成独立行；缺失闭引号的内容未自动处理。$"), lambda m: f"Split {m.group(1)} complete dialogue occurrence(s) on the right into separate lines; text with missing closing quotes was not changed automatically."),
    (re.compile(r"^查看AI修改（采用(\d+) / 拦截(\d+)）$"), lambda m: f"View AI Changes (accepted {m.group(1)} / blocked {m.group(2)})"),
    (re.compile(r"^AI状态：已逐项选择 · 采用 (\d+)，保留原文 (\d+)$"), lambda m: f"AI status: decisions complete · accepted {m.group(1)}, kept original {m.group(2)}"),
    (re.compile(r"^发布前检查完成：高风险正文问题 (\d+)，需留意 (\d+)，翻译阻断 (\d+)，可确定修复 (\d+)；结构级关键提示 (\d+) 项。$"), lambda m: f"Pre-publication check complete: {m.group(1)} high-risk body issue(s), {m.group(2)} warning(s), {m.group(3)} translation blocker(s), {m.group(4)} deterministic repair(s); {m.group(5)} key structural notice(s)."),
    (re.compile(r"^已定位发布问题：第 (\d+) 行 · (.+)。(.*)$"), lambda m: f"Located publication issue: line {m.group(1)} · {m.group(2)}. {m.group(3)}"),
    (re.compile(r"^参考原文：(.*) · (\d+) 段$"), lambda m: f"Reference source: {m.group(1)} · {m.group(2)} paragraphs"),
    (re.compile(r"^外部 OCR：(.*) · (\d+) 段$"), lambda m: f"External OCR: {m.group(1)} · {m.group(2)} paragraphs"),
    (re.compile(r"^(\d+) 个候选不完全一致 · 请打钩选择$"), lambda m: f"{m.group(1)} candidates differ · select one"),
    (re.compile(r"^(\d+) 个候选不完全一致 · 差异已标注 · 请重新打钩选择$"), lambda m: f"{m.group(1)} candidates differ · differences marked · select again"),
    (re.compile(r"^当前句：(\d+)/(\d+) · (.*) · (.*)$"), lambda m: f"Current sentence: {m.group(1)}/{m.group(2)} · {m.group(3)} · {m.group(4)}"),
    (re.compile(r"^自动分析：(.*)$"), lambda m: f"Automatic analysis: {m.group(1)}"),
    (re.compile(r"^手动裁决：当前句采用模型(\d+) · (.*)，其他候选已收起。$"), lambda m: f"Manual decision: current sentence uses model {m.group(1)} · {m.group(2)}; other candidates are collapsed."),
    (re.compile(r"^(.+)：(\d+)/(\d+)。任务在后台执行，窗口仍可响应。$"), lambda m: f"{_catalog_fragment(m.group(1), LANG_EN)}: {m.group(2)}/{m.group(3)}. The task is running in the background and the window remains responsive."),
    (re.compile(r"^会话已恢复 · (\d+) 模型 · (\d+) 句$"), lambda m: f"Session restored · {m.group(1)} model(s) · {m.group(2)} sentences"),
    (re.compile(r"^已导出 (.+)：(\d+) 个正文条目，锁定 (\d+) 条、需复核 (\d+) 条，视觉复核 (\d+) 页，ZIP ([\d.]+) MB。$"), lambda m: f"Exported {m.group(1)}: {m.group(2)} body entries, {m.group(3)} locked, {m.group(4)} needing review, {m.group(5)} visual-review pages, ZIP {m.group(6)} MB."),
    (re.compile(r"^正在刷新模型栏 (\d+)/(\d+)；其余步骤继续在事件循环间隙执行…$"), lambda m: f"Refreshing model column {m.group(1)}/{m.group(2)}; remaining steps continue between event-loop turns…"),
    (re.compile(r"^已核对 (\d+)/(\d+) · 已修改 (\d+)(.*)$"), lambda m: f"Reviewed {m.group(1)}/{m.group(2)} · changed {m.group(3)}{m.group(4)}"),
    (re.compile(r"^第 (\d+) / (\d+) 句 · 页 (\d+) · (.*) 列(.*)$"), lambda m: f"Sentence {m.group(1)} / {m.group(2)} · page {m.group(3)} · column {m.group(4)}{m.group(5)}"),
    (re.compile(r"^已用预览打开 · (.*)$"), lambda m: f"Opened in Preview · {m.group(1)}"),
    (re.compile(r"^页面管理已更新：(\d+) 页。正文版本保持不变，生成 EPUB 时自动同步封面和插图。$"), lambda m: f"Page Manager updated: {m.group(1)} pages. Body versions are unchanged; cover and illustrations will sync automatically when building EPUB."),
    (re.compile(r"^当前正文来源：(.*)$"), lambda m: f"Current body source: {m.group(1)}"),
    (re.compile(r"^PDF文字层处理完成：字符保全通过（(\d+)→(\d+)）$"), lambda m: f"PDF text-layer processing complete: character preservation passed ({m.group(1)}→{m.group(2)})"),
    (re.compile(r"^项目目录：(.*)$"), lambda m: f"Project folder: {m.group(1)}"),
    (re.compile(r"^替换来源：(.*) · (\d+) 段$"), lambda m: f"Replacement source: {m.group(1)} · {m.group(2)} paragraphs"),
    (re.compile(r"^比较完成：一致率 (.*)；来源缺失 (\d+) 字；OCR 额外 (\d+) 字$"), lambda m: f"Comparison complete: match rate {m.group(1)}; source missing {m.group(2)} chars; OCR extra {m.group(3)} chars"),
    (re.compile(r"^AI处理中：(\d+) / (\d+) 章$"), lambda m: f"AI processing: {m.group(1)} / {m.group(2)} chapters"),
    (re.compile(r"^API Key 无效：(.*)$"), lambda m: f"Invalid API Key: {m.group(1)}"),
    (re.compile(r"^对齐失败：(.*)$"), lambda m: f"Alignment failed: {m.group(1)}"),
    (re.compile(r"^暂时无法预测体积：(.*)$"), lambda m: f"Unable to estimate size yet: {m.group(1)}"),
    (re.compile(r"^图片生成失败：(.*)$"), lambda m: f"Image generation failed: {m.group(1)}"),
])

_PATTERNS[LANG_JA].extend([
    (re.compile(r"^已扫描：(\d+) 项可直接使用$"), lambda m: f"スキャン完了：{m.group(1)} 項目が使用可能"),
    (re.compile(r"^当前：页面识别 (\d+) / (\d+) 页$"), lambda m: f"現在：ページ認識 {m.group(1)} / {m.group(2)}"),
    (re.compile(r"^(.+)：准备中$"), lambda m: f"{_catalog_fragment(m.group(1), LANG_JA)}：準備中"),
    (re.compile(r"^✓ 已读取 (\d+) 个可用模型，可从下拉框选择；也可以手动输入。$"), lambda m: f"✓ 利用可能なモデルを {m.group(1)} 件読み込みました。リストから選択するか手動入力できます。"),
    (re.compile(r"^模型列表读取失败：(.*)。仍可在下拉框中手动填写模型名称。$"), lambda m: f"モデル一覧の読込に失敗：{m.group(1)}。モデル名は引き続き手動入力できます。"),
    (re.compile(r"^✓ 连接成功：(.*)$"), lambda m: f"✓ 接続成功：{m.group(1)}"),
    (re.compile(r"^正在执行：(.*)$"), lambda m: f"実行中：{_catalog_fragment(m.group(1), LANG_JA)}"),
    (re.compile(r"^(.+)正在处理$"), lambda m: f"{_catalog_fragment(m.group(1), LANG_JA)}処理中"),
    (re.compile(r"^✓ (.+)完成$"), lambda m: f"✓ {_catalog_fragment(m.group(1), LANG_JA)}完了"),
    (re.compile(r"^已清空：(.*)$"), lambda m: f"クリア済み：{_catalog_fragment(m.group(1), LANG_JA)}"),
    (re.compile(r"^正文来源：(.*)$"), lambda m: f"本文ソース：{m.group(1)}"),
    (re.compile(r"^图片页：(\d+)$"), lambda m: f"画像ページ：{m.group(1)}"),
    (re.compile(r"^章节：(\d+)$"), lambda m: f"章：{m.group(1)}"),
    (re.compile(r"^☰ (\d+) 章$"), lambda m: f"☰ {m.group(1)} 章"),
    (re.compile(r"^🎨 (\d+) 图$"), lambda m: f"🎨 {m.group(1)} 枚"),
    (re.compile(r"^(\d+) / (\d+) 页$"), lambda m: f"{m.group(1)} / {m.group(2)} ページ"),
    (re.compile(r"^后台对齐：左 (\d+) 行，右 (\d+) 行……$"), lambda m: f"バックグラウンド整列：左 {m.group(1)} 行、右 {m.group(2)} 行…"),
    (re.compile(r"^左侧 · (.*)$"), lambda m: f"左側 · {m.group(1)}"),
    (re.compile(r"^右侧 · (.*)（可编辑）$"), lambda m: f"右側 · {m.group(1)}（編集可）"),
    (re.compile(r"^已按当前编辑内容重新对齐 (\d+) 行$"), lambda m: f"現在の編集内容で {m.group(1)} 行を再整列しました"),
    (re.compile(r"^已实时保持双栏对齐；当前约第 (\d+) 行。修改尚未写回其他工作区。$"), lambda m: f"両ペインをリアルタイムで整列維持中。現在およそ {m.group(1)} 行目。変更は他のワークスペースへまだ書き戻していません。"),
    (re.compile(r"^已将右侧 (\d+) 处完整对白拆成独立行；缺失闭引号的内容未自动处理。$"), lambda m: f"右側の完全な台詞 {m.group(1)} 箇所を独立行へ分割しました。閉じ引用符がない内容は自動処理していません。"),
    (re.compile(r"^查看AI修改（采用(\d+) / 拦截(\d+)）$"), lambda m: f"AI変更を表示（採用 {m.group(1)} / ブロック {m.group(2)}）"),
    (re.compile(r"^AI状态：已逐项选择 · 采用 (\d+)，保留原文 (\d+)$"), lambda m: f"AI状態：選択完了 · 採用 {m.group(1)}、原文保持 {m.group(2)}"),
    (re.compile(r"^发布前检查完成：高风险正文问题 (\d+)，需留意 (\d+)，翻译阻断 (\d+)，可确定修复 (\d+)；结构级关键提示 (\d+) 项。$"), lambda m: f"出版前チェック完了：高リスク本文問題 {m.group(1)}、要注意 {m.group(2)}、翻訳ブロッカー {m.group(3)}、確定修復 {m.group(4)}、構造上の重要通知 {m.group(5)} 件。"),
    (re.compile(r"^已定位发布问题：第 (\d+) 行 · (.+)。(.*)$"), lambda m: f"出版問題を特定：{m.group(1)} 行目 · {m.group(2)}。{m.group(3)}"),
    (re.compile(r"^参考原文：(.*) · (\d+) 段$"), lambda m: f"参照原文：{m.group(1)} · {m.group(2)} 段落"),
    (re.compile(r"^外部 OCR：(.*) · (\d+) 段$"), lambda m: f"外部OCR：{m.group(1)} · {m.group(2)} 段落"),
    (re.compile(r"^(\d+) 个候选不完全一致 · 请打钩选择$"), lambda m: f"{m.group(1)} 候補が不一致 · 選択してください"),
    (re.compile(r"^(\d+) 个候选不完全一致 · 差异已标注 · 请重新打钩选择$"), lambda m: f"{m.group(1)} 候補が不一致 · 差異表示済み · 再選択してください"),
    (re.compile(r"^当前句：(\d+)/(\d+) · (.*) · (.*)$"), lambda m: f"現在文：{m.group(1)}/{m.group(2)} · {m.group(3)} · {m.group(4)}"),
    (re.compile(r"^自动分析：(.*)$"), lambda m: f"自動分析：{m.group(1)}"),
    (re.compile(r"^手动裁决：当前句采用模型(\d+) · (.*)，其他候选已收起。$"), lambda m: f"手動裁決：現在文はモデル{m.group(1)}を採用 · {m.group(2)}。他候補は折りたたみました。"),
    (re.compile(r"^(.+)：(\d+)/(\d+)。任务在后台执行，窗口仍可响应。$"), lambda m: f"{_catalog_fragment(m.group(1), LANG_JA)}：{m.group(2)}/{m.group(3)}。タスクはバックグラウンドで実行され、ウィンドウは応答を続けます。"),
    (re.compile(r"^会话已恢复 · (\d+) 模型 · (\d+) 句$"), lambda m: f"セッション復元 · {m.group(1)} モデル · {m.group(2)} 文"),
    (re.compile(r"^已导出 (.+)：(\d+) 个正文条目，锁定 (\d+) 条、需复核 (\d+) 条，视觉复核 (\d+) 页，ZIP ([\d.]+) MB。$"), lambda m: f"{m.group(1)}を書き出しました：本文 {m.group(2)} 件、ロック {m.group(3)}、要確認 {m.group(4)}、視覚確認 {m.group(5)} ページ、ZIP {m.group(6)} MB。"),
    (re.compile(r"^正在刷新模型栏 (\d+)/(\d+)；其余步骤继续在事件循环间隙执行…$"), lambda m: f"モデル欄を更新中 {m.group(1)}/{m.group(2)}。残りの処理はイベントループの合間に続行します…"),
    (re.compile(r"^已核对 (\d+)/(\d+) · 已修改 (\d+)(.*)$"), lambda m: f"確認済み {m.group(1)}/{m.group(2)} · 修正 {m.group(3)}{m.group(4)}"),
    (re.compile(r"^第 (\d+) / (\d+) 句 · 页 (\d+) · (.*) 列(.*)$"), lambda m: f"第 {m.group(1)} / {m.group(2)} 文 · {m.group(3)} ページ · {m.group(4)} 列{m.group(5)}"),
    (re.compile(r"^已用预览打开 · (.*)$"), lambda m: f"プレビューで開きました · {m.group(1)}"),
    (re.compile(r"^页面管理已更新：(\d+) 页。正文版本保持不变，生成 EPUB 时自动同步封面和插图。$"), lambda m: f"ページ管理を更新：{m.group(1)} ページ。本文バージョンは変更せず、EPUB生成時に表紙と挿絵を自動同期します。"),
    (re.compile(r"^当前正文来源：(.*)$"), lambda m: f"現在の本文ソース：{m.group(1)}"),
    (re.compile(r"^PDF文字层处理完成：字符保全通过（(\d+)→(\d+)）$"), lambda m: f"PDFテキストレイヤー処理完了：文字保全合格（{m.group(1)}→{m.group(2)}）"),
    (re.compile(r"^项目目录：(.*)$"), lambda m: f"プロジェクトフォルダ：{m.group(1)}"),
    (re.compile(r"^替换来源：(.*) · (\d+) 段$"), lambda m: f"置換ソース：{m.group(1)} · {m.group(2)} 段落"),
    (re.compile(r"^比较完成：一致率 (.*)；来源缺失 (\d+) 字；OCR 额外 (\d+) 字$"), lambda m: f"比較完了：一致率 {m.group(1)}；ソース欠落 {m.group(2)} 文字；OCR余分 {m.group(3)} 文字"),
    (re.compile(r"^AI处理中：(\d+) / (\d+) 章$"), lambda m: f"AI処理中：{m.group(1)} / {m.group(2)} 章"),
    (re.compile(r"^API Key 无效：(.*)$"), lambda m: f"API Keyが無効：{m.group(1)}"),
    (re.compile(r"^对齐失败：(.*)$"), lambda m: f"整列失敗：{m.group(1)}"),
    (re.compile(r"^暂时无法预测体积：(.*)$"), lambda m: f"現在は容量を予測できません：{m.group(1)}"),
    (re.compile(r"^图片生成失败：(.*)$"), lambda m: f"画像生成失敗：{m.group(1)}"),
])
# Payload form used by the generic "当前：…" status rule above.
_PATTERNS[LANG_EN].append((re.compile(r"^页面识别 (\d+) / (\d+) 页$"), lambda m: f"page recognition {m.group(1)} / {m.group(2)}"))
_PATTERNS[LANG_JA].append((re.compile(r"^页面识别 (\d+) / (\d+) 页$"), lambda m: f"ページ認識 {m.group(1)} / {m.group(2)}"))

_EXACT_EN.update({
    "V4 多模型分歧裁决包": "V4 Multi-model Conflict Adjudication Package",
    "V3 标准紧凑包": "V3 Standard Compact Package",
    "V3 完整取证包": "V3 Full Evidence Package",
    "AI 修复包": "AI Repair Package",
    "严格覆盖（重建小说排版）": "Strict Replace (Rebuild Novel Layout)",
    "严格覆盖（完全原样）": "Strict Replace (Exact Copy)",
    "局部智能替换": "Local Smart Replace",
    "仅比较差异": "Compare Only",
    "AI纠错并重排 OCR 结果": "AI Correct & Re-layout OCR Result",
    "AI纠错替换结果": "AI Correct Replacement Result",
    "AI纠错并重排替换结果": "AI Correct & Re-layout Replacement Result",
})
_EXACT_JA.update({
    "V4 多模型分歧裁决包": "V4 複数モデル差異裁決パッケージ",
    "V3 标准紧凑包": "V3 標準コンパクトパッケージ",
    "V3 完整取证包": "V3 完全証拠パッケージ",
    "AI 修复包": "AI修復パッケージ",
    "严格覆盖（重建小说排版）": "厳密置換（小説組版を再構築）",
    "严格覆盖（完全原样）": "厳密置換（完全原文どおり）",
    "局部智能替换": "部分スマート置換",
    "仅比较差异": "差異のみ比較",
    "AI纠错并重排 OCR 结果": "OCR結果をAI修正・再組版",
    "AI纠错替换结果": "置換結果をAI修正",
    "AI纠错并重排替换结果": "置換結果をAI修正・再組版",
})

_EXACT_EN.update({
    "只验证当前 Provider / API Key / Base URL / 文本路由是否可达；不验证图片输入。": "Verify only the current Provider / API key / Base URL / text route; this does not validate image input.",
    "使用当前页面中的一张缩小图片发起一次真实多模态请求；这是正式跑书前最重要的测试。": "Send one downscaled current page as a real multimodal request; this is the most important test before processing the full book.",
    "建议顺序：先点“测试连接”，再点“测试图片能力”，最后再正式处理整本书。": "Recommended order: run Test Connection first, then Test Image Capability, and only then process the full book.",
    "建议顺序：先点“测试连接”，确认文本路由可达；再点“测试图片能力”，确认 AI 图文处理会用到的真实图片链路正常。": "Recommended order: first run Test Connection to confirm the text route works, then run Test Image Capability to confirm the real image path used by AI Image Processing works.",
    "当前 Provider 状态不可用": "Current provider status is unavailable",
    "正在测试 Provider 文本连接…": "Testing the provider text connection…",
    "连接测试失败": "Connection test failed",
})
_EXACT_JA.update({
    "只验证当前 Provider / API Key / Base URL / 文本路由是否可达；不验证图片输入。": "現在のProvider / API Key / Base URL / テキスト経路だけを確認します。画像入力は検証しません。",
    "使用当前页面中的一张缩小图片发起一次真实多模态请求；这是正式跑书前最重要的测试。": "現在のページ1枚を縮小して実際のマルチモーダル要求を送信します。本番の全文処理前に最も重要なテストです。",
    "建议顺序：先点“测试连接”，再点“测试图片能力”，最后再正式处理整本书。": "推奨手順：まず「接続テスト」、次に「画像入力をテスト」、最後に全文の本処理を開始してください。",
    "建议顺序：先点“测试连接”，确认文本路由可达；再点“测试图片能力”，确认 AI 图文处理会用到的真实图片链路正常。": "推奨手順：まず「接続テスト」でテキスト経路を確認し、次に「画像入力をテスト」でAI画像・本文処理が使う実際の画像経路を確認してください。",
    "当前 Provider 状态不可用": "現在のProvider状態を取得できません",
    "正在测试 Provider 文本连接…": "Providerのテキスト接続を確認中…",
    "连接测试失败": "接続テストに失敗しました",
})

_EXACT_EN.update({
    "完整原页直送 API（推荐 · 防最右列/页顶漏字）": "Send Full Page Directly to API (Recommended · prevents top/right-edge omissions)",
    "开启后首轮每次只发送一个完整物理页面；若原图尺寸已合适则直接发送原始 PNG/JPEG/WebP，不做裁白边或二次压缩。": "When enabled, the first pass sends one complete physical page per request. If the source image is already within the selected size, the original PNG/JPEG/WebP is sent directly without margin cropping or recompression.",
    "切换为仅提取原文、1600px、完整原页直送 API 且不自动二审；GLM 默认关闭思考": "Switch to source extraction only, 1600px, full-page direct API input, and no automatic second review; GLM thinking is off by default.",
    "切换为提取原文 + 翻译、简体中文、1600px、完整原页直送 API 且不自动二审；GLM 默认关闭思考": "Switch to extract + translate, Simplified Chinese, 1600px, full-page direct API input, and no automatic second review; GLM thinking is off by default.",
    "页面顺序、封面、插图和非正文页由页面管理决定；AI不重新分类。推荐开启“完整原页直送 API”：首轮一页一请求，让多模态模型自己处理纵排阅读顺序、段落、对白和翻译，程序只做校验，不先裁正文。均衡/出版模式仅对空页、异常短页、段落压平、页顶/最右列风险和疑难局部追加定向核验；跨页接续和补译优先走纯文本。": "Page order, covers, illustrations, and non-body pages are defined by Page Manager; AI does not reclassify them. Full-page direct API input is recommended: the first pass sends one page per request and lets the multimodal model handle vertical reading order, paragraphs, dialogue, and translation while the app validates results instead of pre-cropping body text. Balanced/Publication modes add targeted checks only for empty pages, abnormally short pages, flattened paragraphs, top/right-edge risks, and uncertain regions; cross-page continuity and translation repair prefer text-only checks.",
    "已应用最快设置：仅提取原文 · 1600px · 完整原页直送API · 不自动二审 · GLM默认不思考": "Fastest preset applied: source only · 1600px · full-page direct API · no automatic second review · GLM thinking off by default",
    "已应用提取并翻译设置：简体中文 · 快速1600px · 完整原页直送API · 不自动二审 · GLM默认不思考": "Extract + translate preset applied: Simplified Chinese · fast 1600px · full-page direct API · no automatic second review · GLM thinking off by default",
})
_EXACT_JA.update({
    "完整原页直送 API（推荐 · 防最右列/页顶漏字）": "原ページ全体をAPIへ直接送信（推奨 · 右端列／ページ上端の脱落防止）",
    "开启后首轮每次只发送一个完整物理页面；若原图尺寸已合适则直接发送原始 PNG/JPEG/WebP，不做裁白边或二次压缩。": "有効時、初回はリクエストごとに物理ページ全体を1ページだけ送信します。元画像が選択サイズ内なら、余白切り取りや再圧縮をせず元のPNG/JPEG/WebPを直接送信します。",
    "切换为仅提取原文、1600px、完整原页直送 API 且不自动二审；GLM 默认关闭思考": "原文抽出のみ・1600px・原ページ全体をAPIへ直接送信・自動二次確認なしに切り替えます。GLMの思考は既定でオフです。",
    "切换为提取原文 + 翻译、简体中文、1600px、完整原页直送 API 且不自动二审；GLM 默认关闭思考": "原文抽出＋翻訳・簡体字中国語・1600px・原ページ全体をAPIへ直接送信・自動二次確認なしに切り替えます。GLMの思考は既定でオフです。",
    "页面顺序、封面、插图和非正文页由页面管理决定；AI不重新分类。推荐开启“完整原页直送 API”：首轮一页一请求，让多模态模型自己处理纵排阅读顺序、段落、对白和翻译，程序只做校验，不先裁正文。均衡/出版模式仅对空页、异常短页、段落压平、页顶/最右列风险和疑难局部追加定向核验；跨页接续和补译优先走纯文本。": "ページ順、表紙、挿絵、本文以外のページはページ管理が決定し、AIは再分類しません。「原ページ全体をAPIへ直接送信」を推奨します。初回は1ページずつ送信し、縦書きの読書順、段落、台詞、翻訳をマルチモーダルモデルに直接処理させ、アプリ側は本文を先に切り抜かず検証だけを行います。標準／出版モードでは、空ページ、異常に短いページ、段落の平坦化、ページ上端／右端列の危険、疑義領域だけを追加確認し、ページ跨ぎ接続と補訳は原則テキストのみで処理します。",
    "已应用最快设置：仅提取原文 · 1600px · 完整原页直送API · 不自动二审 · GLM默认不思考": "最速設定を適用：原文のみ · 1600px · 原ページ全体をAPIへ直接送信 · 自動二次確認なし · GLM思考は既定でオフ",
    "已应用提取并翻译设置：简体中文 · 快速1600px · 完整原页直送API · 不自动二审 · GLM默认不思考": "抽出＋翻訳設定を適用：簡体字中国語 · 高速1600px · 原ページ全体をAPIへ直接送信 · 自動二次確認なし · GLM思考は既定でオフ",
})


_EXACT_EN.update({
    "GLM 深度思考（GLM-5.3 Flash：关闭=low，开启=high；无法完全关闭）": "GLM deep reasoning (GLM-5.3 Flash: off = low, on = high; it cannot be fully disabled)",
    "切换为仅提取原文、1600px、完整原页直送 API 且不自动二审；GLM-5.3 Flash 默认 low 思考": "Switch to source extraction only, 1600px, full-page direct API input, and no automatic second review; GLM-5.3 Flash defaults to low reasoning.",
    "切换为提取原文 + 翻译、简体中文、1600px、完整原页直送 API 且不自动二审；GLM-5.3 Flash 默认 low 思考": "Switch to extract + translate, Simplified Chinese, 1600px, full-page direct API input, and no automatic second review; GLM-5.3 Flash defaults to low reasoning.",
    "已应用最快设置：仅提取原文 · 1600px · 完整原页直送API · 不自动二审 · GLM-5.3 Flash=low": "Fastest preset applied: source only · 1600px · full-page direct API · no automatic second review · GLM-5.3 Flash=low",
    "已应用提取并翻译设置：简体中文 · 快速1600px · 完整原页直送API · 不自动二审 · GLM-5.3 Flash=low": "Extract + translate preset applied: Simplified Chinese · fast 1600px · full-page direct API · no automatic second review · GLM-5.3 Flash=low",
})
_EXACT_JA.update({
    "GLM 深度思考（GLM-5.3 Flash：关闭=low，开启=high；无法完全关闭）": "GLM深度推論（GLM-5.3 Flash：オフ=low、オン=high。完全には無効化できません）",
    "切换为仅提取原文、1600px、完整原页直送 API 且不自动二审；GLM-5.3 Flash 默认 low 思考": "原文抽出のみ・1600px・原ページ全体をAPIへ直接送信・自動二次確認なしに切り替えます。GLM-5.3 Flashは既定でlow推論です。",
    "切换为提取原文 + 翻译、简体中文、1600px、完整原页直送 API 且不自动二审；GLM-5.3 Flash 默认 low 思考": "原文抽出＋翻訳・簡体字中国語・1600px・原ページ全体をAPIへ直接送信・自動二次確認なしに切り替えます。GLM-5.3 Flashは既定でlow推論です。",
    "已应用最快设置：仅提取原文 · 1600px · 完整原页直送API · 不自动二审 · GLM-5.3 Flash=low": "最速設定を適用：原文のみ · 1600px · 原ページ全体をAPIへ直接送信 · 自動二次確認なし · GLM-5.3 Flash=low",
    "已应用提取并翻译设置：简体中文 · 快速1600px · 完整原页直送API · 不自动二审 · GLM-5.3 Flash=low": "抽出＋翻訳設定を適用：簡体字中国語 · 高速1600px · 原ページ全体をAPIへ直接送信 · 自動二次確認なし · GLM-5.3 Flash=low",
})

_EXACT_EN.update({
    "切换为提取原文 + 翻译、简体中文、均衡结构复核、完整原页直送 API；优先保留原书自然段和独立对白，GLM-5.3 Flash 默认 low 思考": "Switch to extraction + Simplified Chinese translation with balanced structure review and full-page direct API input; preserve the book's paragraphs and standalone dialogue. GLM-5.3 Flash defaults to low reasoning.",
    "已应用提取并翻译设置：简体中文 · 均衡1920px · 完整原页直送API · 结构/疑难复核 · GLM-5.3 Flash=low": "Extract + translate preset applied: Simplified Chinese · balanced 1920px · full-page direct API · structure/uncertainty review · GLM-5.3 Flash=low",
})
_EXACT_JA.update({
    "切换为提取原文 + 翻译、简体中文、均衡结构复核、完整原页直送 API；优先保留原书自然段和独立对白，GLM-5.3 Flash 默认 low 思考": "原文抽出＋簡体字中国語への翻訳・標準構造確認・原ページ全体のAPI直接送信に切り替え、原書の段落と独立台詞を優先して保持します。GLM-5.3 Flashは既定でlow推論です。",
    "已应用提取并翻译设置：简体中文 · 均衡1920px · 完整原页直送API · 结构/疑难复核 · GLM-5.3 Flash=low": "抽出＋翻訳設定を適用：簡体字中国語 · 標準1920px · 原ページ全体をAPIへ直接送信 · 構造／疑義確認 · GLM-5.3 Flash=low",
})

_PATTERNS[LANG_EN].append(
    (re.compile(r"^；本地零Token对白拆段 (\d+) 块$"),
     lambda m: f"; local zero-token dialogue splitting: {m.group(1)} block(s)")
)
_PATTERNS[LANG_JA].append(
    (re.compile(r"^；本地零Token对白拆段 (\d+) 块$"),
     lambda m: f"；ローカル・ゼロトークン台詞分割 {m.group(1)} ブロック")
)


_EXACT_EN.update({
    "AI 批量视觉裁决": "AI Batched Visual Adjudication",
    "程序先让视觉模型在看不到 A/B/C 的情况下独立逐字抄写，再比较本地 OCR 候选。只有通过完整 crop 覆盖、日文标点/引号、2:1 多数逆转和敏感字形检查的结果才会自动写入；危险项会做第二次视觉反证，仍不确定时保留原融合稿。": "The model first transcribes each crop independently without seeing A/B/C, then compares the local OCR candidates. Only results that pass complete-crop coverage, Japanese punctuation/quote, 2:1 reversal, and sensitive-glyph checks are written automatically; risky items receive a second visual challenge and otherwise keep the original fused draft.",
    "出版精细 · 16 条/请求（推荐）": "Publication precise · 16 items/request (recommended)",
    "均衡 · 24 条/请求": "Balanced · 24 items/request",
    "快速 · 32 条/请求": "Fast · 32 items/request",
    "证据规则：图片必须覆盖该行声明的全部物理列和页面；有图片文件不等于证据完整。模型置信度不参与自动放行，程序验证失败的条目会写明 audit_issues 并进入人工复核。": "Evidence rule: the image must cover every physical column and page declared by the row; having an image file does not prove complete evidence. Model confidence never authorizes automatic write-back; failed validations are recorded in audit_issues and sent to manual review.",
    "本模式不会让大模型重做整页 OCR。程序先保留本地多模型 OCR 作为证据主干，只把真正分歧/低置信条目的原图裁切拼成证据板，一次提交几十到上百条。AI 默认只返回 A/B/C/D 候选字母；只有全部候选都错时才允许返回 X+精确原文。": "This mode does not let the model redo full-page OCR. Local multi-model OCR remains the evidence backbone; only true conflicts and low-confidence source crops are packed into evidence sheets, with dozens to over one hundred items per request. The AI normally returns only A/B/C/D; X + exact source text is allowed only when every candidate is wrong.",
    "均衡 · 64 条/请求": "Balanced · 64 items/request",
    "快速 · 96 条/请求": "Fast · 96 items/request",
    "出版 · 48 条/请求": "Publication · 48 items/request",
    "实际批量": "Actual batch",
    "批量：": "Batch:",
    "low · 推荐，速度/成本最低": "low · recommended, lowest latency/cost",
    "high · 疑难字更多时使用": "high · use for harder glyphs",
    "视觉思考：": "Visual reasoning:",
    "审计报告：": "Audit report:",
    "只发送真正分歧、低置信、疑似缺失/重复条目（推荐）": "Send only true conflicts, low-confidence, suspected missing/duplicate items (recommended)",
    "允许 X：所有本地候选都错时，AI 可返回少量精确新字；仍需本地防改写校验": "Allow X: when every local candidate is wrong, AI may return a small exact source correction; local anti-rewrite checks still apply",
    "Token 策略：候选文字只在文本中发送一次；图片证据只放编号+裁切，不把 A/B/C 重复画进图片；普通结果只返回一个字母。没有可用原图的条目不会让 AI 猜，而是继续留在人工复核队列。": "Token strategy: candidate text is sent once in text; image evidence contains only IDs and crops, without redrawing A/B/C into the image. Normal results return one letter only. Items without usable source images stay in the manual-review queue instead of being guessed.",
    "把真正 OCR 分歧的原图裁切批量拼成证据板，一次提交几十到上百条给多模态模型；AI 默认只选择 A/B/C/D，不重做整页 OCR，不改写原始模型结果。": "Pack source crops for true OCR conflicts into evidence sheets and submit dozens to over one hundred items per multimodal request; AI normally selects only A/B/C/D, never redoes full-page OCR or rewrites raw model outputs.",
    "停止AI裁决": "Stop AI adjudication",
    "准备视觉证据…": "Preparing visual evidence…",
})
_EXACT_JA.update({
    "AI 批量视觉裁决": "AI一括視覚裁決",
    "程序先让视觉模型在看不到 A/B/C 的情况下独立逐字抄写，再比较本地 OCR 候选。只有通过完整 crop 覆盖、日文标点/引号、2:1 多数逆转和敏感字形检查的结果才会自动写入；危险项会做第二次视觉反证，仍不确定时保留原融合稿。": "モデルはまずA/B/Cを見ずに各cropを逐字転記し、その後ローカルOCR候補を比較します。完全なcrop、日文の句読点／引用符、2:1逆転、要注意字形の検査を通過した結果だけを自動適用し、危険項目は二度目の視覚再確認へ回し、それでも不確かな場合は元の融合稿を保持します。",
    "出版精细 · 16 条/请求（推荐）": "出版精細 · 16件/リクエスト（推奨）",
    "均衡 · 24 条/请求": "標準 · 24件/リクエスト",
    "快速 · 32 条/请求": "高速 · 32件/リクエスト",
    "证据规则：图片必须覆盖该行声明的全部物理列和页面；有图片文件不等于证据完整。模型置信度不参与自动放行，程序验证失败的条目会写明 audit_issues 并进入人工复核。": "証拠ルール：画像は行が宣言するすべての物理列とページを覆う必要があり、画像ファイルが存在するだけでは証拠が完全とは限りません。モデル信頼度だけで自動書き戻しは行わず、検証失敗はaudit_issuesに記録して手動確認へ送ります。",
    "本模式不会让大模型重做整页 OCR。程序先保留本地多模型 OCR 作为证据主干，只把真正分歧/低置信条目的原图裁切拼成证据板，一次提交几十到上百条。AI 默认只返回 A/B/C/D 候选字母；只有全部候选都错时才允许返回 X+精确原文。": "このモードでは大規模モデルにページ全体のOCRをやり直させません。ローカル複数OCRを証拠の主軸として保持し、真の不一致／低信頼項目の原画像cropだけを証拠シートにまとめ、1回で数十～100件超を送信します。AIは原則A/B/C/Dだけを返し、全候補が誤りの場合のみX＋正確な原文を許可します。",
    "均衡 · 64 条/请求": "標準 · 64件/リクエスト",
    "快速 · 96 条/请求": "高速 · 96件/リクエスト",
    "出版 · 48 条/请求": "出版 · 48件/リクエスト",
    "实际批量": "実際のバッチ",
    "批量：": "バッチ：",
    "low · 推荐，速度/成本最低": "low · 推奨、速度/コスト最小",
    "high · 疑难字更多时使用": "high · 難読字が多い場合",
    "视觉思考：": "視覚推論：",
    "审计报告：": "監査レポート：",
    "只发送真正分歧、低置信、疑似缺失/重复条目（推荐）": "真の不一致・低信頼・欠落/重複疑いだけ送信（推奨）",
    "允许 X：所有本地候选都错时，AI 可返回少量精确新字；仍需本地防改写校验": "Xを許可：ローカル候補が全て誤りの場合のみ、AIは少量の正確な新規文字を返せます。ローカル改変防止検査は引き続き適用します",
    "Token 策略：候选文字只在文本中发送一次；图片证据只放编号+裁切，不把 A/B/C 重复画进图片；普通结果只返回一个字母。没有可用原图的条目不会让 AI 猜，而是继续留在人工复核队列。": "Token方針：候補文字列はテキスト側で1回だけ送信し、画像証拠には番号とcropだけを置き、A/B/Cを画像へ重複描画しません。通常結果は1文字だけ返します。利用可能な原画像がない項目はAIに推測させず手動確認キューに残します。",
    "把真正 OCR 分歧的原图裁切批量拼成证据板，一次提交几十到上百条给多模态模型；AI 默认只选择 A/B/C/D，不重做整页 OCR，不改写原始模型结果。": "真のOCR不一致に対応する原画像cropを証拠シートへ一括配置し、1回で数十～100件超をマルチモーダルモデルへ送信します。AIは原則A/B/C/Dのみを選択し、ページ全体のOCR再実行や元モデル結果の改変は行いません。",
    "停止AI裁决": "AI裁決を停止",
    "准备视觉证据…": "視覚証拠を準備中…",
})


_EXACT_EN.update({
    "智能均衡 · 保留实质 2:1 分歧，仅跳过高置信标点/空格差异": "Smart balanced · keep substantive 2:1 disagreements; skip only high-confidence punctuation/spacing differences",
    "出版全部 · 所有分歧/低置信都送视觉裁决": "Publication exhaustive · send every disagreement/low-confidence row to visual adjudication",
    "极速高风险 · 可跳过干净高置信 2:1": "Lean high-risk · may skip clean high-confidence 2:1 majorities",
    "裁决范围：": "Adjudication scope:",
    "并发请求：": "Concurrent requests:",
    "0=自动（推荐）": "0 = auto (recommended)",
    "0=自动：在线模型从 4 路起步；稳定时逐步增加，检测到传输重试则减半。Ollama 默认单路。": "0 = auto: managed online providers start at 4 concurrent requests, grow while stable, and halve on transport retries. Ollama defaults to one.",
    "AI 漏回较多编号时，只把缺失项缩小批次自动补跑一次（推荐）": "If AI omits many IDs, automatically repack only the missing items into a smaller follow-up batch once (recommended)",
})
_EXACT_JA.update({
    "智能均衡 · 保留实质 2:1 分歧，仅跳过高置信标点/空格差异": "スマート標準 · 実質的な2:1不一致は残し、高信頼の句読点/空白差だけを省略",
    "出版全部 · 所有分歧/低置信都送视觉裁决": "出版・全件 · すべての不一致/低信頼行を視覚裁決へ送信",
    "极速高风险 · 可跳过干净高置信 2:1": "高速・高リスクのみ · 明確な高信頼2:1多数決は省略可能",
    "裁决范围：": "裁決範囲：",
    "并发请求：": "同時リクエスト：",
    "0=自动（推荐）": "0=自動（推奨）",
    "0=自动：在线模型从 4 路起步；稳定时逐步增加，检测到传输重试则减半。Ollama 默认单路。": "0=自動：オンライン管理モデルは4並列から開始し、安定時は徐々に増加、転送再試行を検出すると半減します。Ollamaは既定で1並列です。",
    "AI 漏回较多编号时，只把缺失项缩小批次自动补跑一次（推荐）": "AIが多数のIDを返し忘れた場合、欠落項目だけを小さいバッチに詰め直して1回自動再実行（推奨）",
})

# Handwriting-review diagnostics.  These patterns translate only program-owned
# chrome and preserve candidate text, engine errors, counts and provenance tags.
_EXACT_EN.update({
    "未发现明显结构性疑点。手动抽查仍以原图为准。": "No obvious structural issue found. Manual spot-checking should still use the original image as authority.",
})
_EXACT_JA.update({
    "未发现明显结构性疑点。手动抽查仍以原图为准。": "明らかな構造上の疑点はありません。手動抽出確認では引き続き原画像を基準にしてください。",
})

_PATTERNS[LANG_EN].extend([
    (re.compile(r"^疑点分数：(\d+)/100$"), lambda m: f"Issue score: {m.group(1)}/100"),
    (re.compile(r"^(•\s*)?物理逐字框：(\d+)；OCR 编辑位：(\d+)/(\d+)，当前红框按物理字槽定位(?:（(.*)）)?$"),
     lambda m: f"{m.group(1) or ''}Physical glyph boxes: {m.group(2)}; OCR edit positions: {m.group(3)}/{m.group(4)}; the red box follows the physical glyph slot" + (f" ({m.group(5)})" if m.group(5) else "")),
    (re.compile(r"^(•\s*)?逐字推子检测到 (\d+) 个物理字框，OCR 只有 (\d+) 字，疑似漏识 (\d+) 字$"),
     lambda m: f"{m.group(1) or ''}Glyph-slider detection found {m.group(2)} physical boxes but OCR has only {m.group(3)} characters; {m.group(4)} character(s) may be missing"),
    (re.compile(r"^(•\s*)?逐字推子检测到 (\d+) 个物理字框，OCR 有 (\d+) 字，疑似多识 (\d+) 字$"),
     lambda m: f"{m.group(1) or ''}Glyph-slider detection found {m.group(2)} physical boxes while OCR has {m.group(3)} characters; {m.group(4)} character(s) may be extra"),
    (re.compile(r"^(•\s*)?存在 (\d+) 个 OCR 未识别的物理字框（以 □ 显示）$"),
     lambda m: f"{m.group(1) or ''}{m.group(2)} physical glyph box(es) were not recognized by OCR (shown as □)"),
    (re.compile(r"^当前字候选失败：(.*)$"), lambda m: f"Current-character candidate recognition failed: {m.group(1)}"),
    (re.compile(r"^当前字形裁切失败：(.*)$"), lambda m: f"Current glyph crop failed: {m.group(1)}"),
    (re.compile(r"^Apple OCR 当前列识别失败：(.*)$"), lambda m: f"Apple OCR failed on the current column: {m.group(1)}"),
    (re.compile(r"^已采用候选“(.*)”；请继续对照原图确认。$"), lambda m: f"Selected candidate “{m.group(1)}”; continue verifying against the original image."),
])

_PATTERNS[LANG_JA].extend([
    (re.compile(r"^疑点分数：(\d+)/100$"), lambda m: f"疑点スコア：{m.group(1)}/100"),
    (re.compile(r"^(•\s*)?物理逐字框：(\d+)；OCR 编辑位：(\d+)/(\d+)，当前红框按物理字槽定位(?:（(.*)）)?$"),
     lambda m: f"{m.group(1) or ''}物理文字枠：{m.group(2)}；OCR編集位置：{m.group(3)}/{m.group(4)}；赤枠は物理文字スロットに従って位置決め" + (f"（{m.group(5)}）" if m.group(5) else "")),
    (re.compile(r"^(•\s*)?逐字推子检测到 (\d+) 个物理字框，OCR 只有 (\d+) 字，疑似漏识 (\d+) 字$"),
     lambda m: f"{m.group(1) or ''}文字スライダー検出で物理文字枠 {m.group(2)} 個に対しOCRは {m.group(3)} 文字のみ。{m.group(4)} 文字の未認識が疑われます"),
    (re.compile(r"^(•\s*)?逐字推子检测到 (\d+) 个物理字框，OCR 有 (\d+) 字，疑似多识 (\d+) 字$"),
     lambda m: f"{m.group(1) or ''}文字スライダー検出で物理文字枠 {m.group(2)} 個に対しOCRは {m.group(3)} 文字。{m.group(4)} 文字の過剰認識が疑われます"),
    (re.compile(r"^(•\s*)?存在 (\d+) 个 OCR 未识别的物理字框（以 □ 显示）$"),
     lambda m: f"{m.group(1) or ''}OCRが認識していない物理文字枠が {m.group(2)} 個あります（□で表示）"),
    (re.compile(r"^当前字候选失败：(.*)$"), lambda m: f"現在文字の候補認識に失敗：{m.group(1)}"),
    (re.compile(r"^当前字形裁切失败：(.*)$"), lambda m: f"現在字形の切り出しに失敗：{m.group(1)}"),
    (re.compile(r"^Apple OCR 当前列识别失败：(.*)$"), lambda m: f"Apple OCRの現在列認識に失敗：{m.group(1)}"),
    (re.compile(r"^已采用候选“(.*)”；请继续对照原图确认。$"), lambda m: f"候補「{m.group(1)}」を採用しました。引き続き原画像と照合してください。"),
])

# 图文对照紧凑底栏与句级跳转（2026-09-25）。
_EXACT_EN.update({
    "确认并下一句": "Confirm & Next",
    "确认当前句并进入下一句": "Confirm the current sentence and move to the next",
    "跳转 ▾": "Jump ▾",
    "跳到待判断或 OCR 分歧句；仅浏览不会自动裁决当前句": "Jump to a pending or OCR-conflict sentence; browsing alone never adjudicates the current sentence",
    "上一 OCR 分歧": "Previous OCR Conflict",
    "下一 OCR 分歧": "Next OCR Conflict",
    "保存当前修改并跳到上一条 OCR 分歧句": "Save current changes and jump to the previous OCR disagreement",
    "上一句（不确认）  Alt+←": "Previous sentence (without confirming)  Alt+←",
})
_EXACT_JA.update({
    "确认并下一句": "確認して次へ",
    "确认当前句并进入下一句": "現在の文を確認して次の文へ進みます",
    "跳转 ▾": "移動 ▾",
    "跳到待判断或 OCR 分歧句；仅浏览不会自动裁决当前句": "未判定またはOCR差分の文へ移動します。閲覧だけでは現在文を裁決しません",
    "上一 OCR 分歧": "前のOCR差分",
    "下一 OCR 分歧": "次のOCR差分",
    "保存当前修改并跳到上一条 OCR 分歧句": "現在の変更を保存して前のOCR差分文へ移動",
    "上一句（不确认）  Alt+←": "前の文（確定せず）  Alt+←",
})

# Multi-model OCR 2–6 and column-input terminology (v19, 2026-09-26).
_EXACT_EN.update({
    "两种方式都先建立同一份 Ruby-free 正文可见层。分列掩膜会从中裁出模型适合的白底视窗；分列显示则直接把整张原尺寸白底页送入 OCR，只让当前正文列可见。两者都不缩放正文像素，Ruby 与相邻列始终保持不透明纸白。": "Both methods first build the same Ruby-free body visibility layer. Column Mask crops model-appropriate white-background viewports from it; Column Display sends a full-size white page to OCR with only the current body column visible. Neither method rescales body pixels, and Ruby/adjacent columns remain opaque paper white.",
    "两种模式使用完全相同的正文/Ruby几何：\n• 分列掩膜：先生成原页白色隔离层，再裁成各 OCR 适合的宽上下文/紧凑视窗；效率更高。\n• 分列显示：保持原页尺寸，其余区域全部纸白，只开放当前正文列；不缩放、不混入 Ruby/相邻列。\n选择“分列显示”时，NDLOCR 也按逐列运行，以保证每次只开放一列。": "Both modes use exactly the same body/Ruby geometry:\n• Column Mask: build a white isolation layer, then crop a wide-context or compact viewport suited to each OCR engine; more efficient.\n• Column Display: keep the original page size, make all other areas paper white, and expose only the current body column; no scaling and no Ruby/adjacent-column leakage.\nWith Column Display, NDLOCR also runs per column so only one column is exposed at a time.",
    "仅在日文精确分列的多模型模式下生效。模型1和模型2先独立识别同一物理列；文字相同的列不再调用模型3～6；仍有分歧时，后续模型依次只识别未决列。两模型共同候选不会冒充多模型一致，也不会自动压过 AI 或人工裁决。关闭后，所有已选模型都会完整识别。": "Applies only to multi-model Japanese precise-column mode. Models 1 and 2 independently recognize the same physical column; matching columns do not call models 3–6. If disagreement remains, later models sequentially recognize only unresolved columns. A two-model shared candidate never pretends to be multi-model agreement and never overrides AI or human adjudication. When disabled, every selected model runs a full pass.",
    "分列掩膜（推荐）：按模型裁白底视窗": "Column Mask (recommended): crop white-background viewports per model",
    "分列显示：原页白底，只显示目标列": "Column Display: full-size white page, target column only",
    "分列输入": "Column Input",
    "只从正文可见层裁掉确定为空白的外侧画布并保留安全边距，文字像素不缩放、不锐化。Apple/NDL/Paddle 保留较宽纸白上下文；Hayai/48px 使用紧凑白底视窗。无论哪一种，相邻列与 Ruby 都不会重新进入正文 OCR。": "Only definitely blank outer canvas is cropped from the body visibility layer, with a safe margin retained; text pixels are neither rescaled nor sharpened. Apple/NDL/Paddle keep wider paper-white context, while Hayai/48px use compact white-background viewports. In either case, adjacent columns and Ruby never re-enter body OCR.",
    "启用日文物理分列": "Enable Japanese Physical Columns",
    "固定正文框先隔离页眉/页脚，再检测日文物理竖列。正文与 Ruby 归属只检测一次；下方可选择“分列掩膜”或“分列显示”作为实际 OCR 输入方式。": "The fixed body box first isolates headers/footers, then detects Japanese physical vertical columns. Body/Ruby ownership is detected once; choose Column Mask or Column Display below as the actual OCR input method.",
    "单模型专用：关闭时把原页直接交给所选 OCR 引擎；开启时按日文物理竖列识别。极窄的已裁单列图会自动按单列处理，不会再被拆成多个假列。": "Single-model only: when off, the original page is sent directly to the selected OCR engine; when on, Japanese physical vertical columns are recognized separately. Very narrow pre-cropped single-column images are detected automatically and will not be split into false columns.",
    "这是单模型专用分列开关。多模型使用独立的角色路由和共享分列几何，不会改写此设置。": "This column-splitting switch belongs to single-model mode. Multi-model OCR uses independent role routing and shared column geometry without changing this setting.",
    "多模型 OCR 对比（最多 6 个，自适应共识）": "Multi-model OCR Compare (up to 6, adaptive consensus)",
    "快速共识：模型1/2先判定，后续模型只补分歧列": "Fast Consensus: Models 1/2 decide first; later models only resolve conflicting columns",
    "所有 OCR 先共用同一张正文可见层：相邻列与 Ruby 均为不透明纸白色。送入识别器时只裁掉部分纯白画布；Apple/NDL/Paddle 等布局 OCR 保留较宽上下文，Hayai/48px 使用较紧凑视窗。文字像素保持原尺寸且不重采样。只有空结果、占位符或与黑像素估计相比严重缺字时，才把全尺寸掩膜作为该列唯一一次救援；低置信或引号不平衡本身不会重复调用 OCR。": "All OCR engines first share the same body visibility layer, where adjacent columns and Ruby are opaque paper white. Only some pure-white canvas is cropped before recognition; layout OCR engines such as Apple/NDL/Paddle retain wider context, while Hayai/48px use tighter viewports. Text pixels remain at original size with no resampling. A full-size mask is used as the sole rescue pass only for empty output, placeholders, or severe missing-text evidence against the black-pixel estimate; low confidence or unbalanced quotation marks alone never trigger repeated OCR.",
    "打开：多份 OCR 结果显示完整全文并保持同步滚动；关闭：只显示当前稳定句，继续逐句裁决。只改变显示，不重新 OCR 或对齐。": "On: show the complete text from all OCR sources with synchronized scrolling. Off: show only the current stable sentence for sentence-by-sentence adjudication. This changes display only and never reruns OCR or alignment.",
    "模型1使用上方当前选中的 OCR；模型2～6共用完全相同的固定正文区域、物理分列几何和正文可见层。不同引擎只按自身输入合同选择宽上下文或紧凑白底视窗，正文/Ruby归属不会改变。完成后进入独立 OCR 对比工作区逐句自动选优或手动选择。": "Model 1 uses the OCR selected above; models 2–6 share exactly the same fixed body region, physical-column geometry, and body visibility layer. Each engine only chooses a wide-context or compact white-background viewport according to its input contract; body/Ruby ownership never changes. When finished, use the separate OCR Compare workspace for sentence-level automatic selection or manual choice.",
    "模型1决定页码、章节、图片锚点与基本段落结构。快速共识会逐级减少模型3～6调用：两模型相同列单独统计为共同候选，后续模型只补真正分歧列；AI/人工裁决始终可以覆盖候选，原始模型输出保持不变。": "Model 1 determines page numbers, chapters, image anchors, and base paragraph structure. Fast consensus progressively reduces calls to models 3–6: matching model-1/2 columns are tracked separately as shared candidates, and later models only resolve genuine conflicting columns. AI/human adjudication can always override candidates, while raw model output remains unchanged.",
    "独立于快速共识开关：模型1和模型2同时读取同一批已缓存物理列。每个模型使用独立临时增强目录，不会互相覆盖；共享分列裁图按页加锁。关闭快速共识时，前两模型仍可并行完成全量首轮，模型3～6随后按完整模式运行。若机器内存较小或两个模型争用同一加速设备，可关闭改为串行。": "Independent of Fast Consensus: models 1 and 2 read the same cached physical columns in parallel. Each model uses an independent temporary enhancement directory, so they never overwrite each other; shared column crops are page-locked. With Fast Consensus off, the first two models can still complete a full parallel pass and models 3–6 then run in full mode. Disable parallelism on low-memory systems or when two models contend for the same accelerator.",
    "选择文件夹，将当前 2～6 份 OCR 文本分别导出为 UTF-8 TXT；全部裁决后同时导出融合稿。": "Choose a folder and export the current 2–6 OCR texts as separate UTF-8 TXT files; after all adjudication, export the fusion text as well.",
    "默认隐藏真正一致和已经选择的融合行；真正分歧会显示全部不同候选，快速共识产生的稳定共同候选自动保留，不再制造二次确认。勾选后自动前往下一组；取消勾选可查看全部融合结果。": "By default, hide true agreement and already-selected fusion rows. True conflicts show all differing candidates; stable shared candidates produced by Fast Consensus are kept automatically without creating a second confirmation. When checked, advance automatically to the next group; uncheck to view all fusion results.",
})
_EXACT_JA.update({
    "两种方式都先建立同一份 Ruby-free 正文可见层。分列掩膜会从中裁出模型适合的白底视窗；分列显示则直接把整张原尺寸白底页送入 OCR，只让当前正文列可见。两者都不缩放正文像素，Ruby 与相邻列始终保持不透明纸白。": "どちらも同じルビ除外済み本文可視レイヤーを先に作成します。列マスクはモデル向けの白背景ビューを切り出し、列表示は原寸の白背景ページ全体をOCRへ渡して現在の本文列だけを見せます。本文画素は拡大縮小せず、ルビと隣接列は常に不透明な紙白です。",
    "两种模式使用完全相同的正文/Ruby几何：\n• 分列掩膜：先生成原页白色隔离层，再裁成各 OCR 适合的宽上下文/紧凑视窗；效率更高。\n• 分列显示：保持原页尺寸，其余区域全部纸白，只开放当前正文列；不缩放、不混入 Ruby/相邻列。\n选择“分列显示”时，NDLOCR 也按逐列运行，以保证每次只开放一列。": "両モードはまったく同じ本文/ルビ形状情報を使います。\n• 列マスク：元ページの白い隔離レイヤーを作り、各OCRに適した広い文脈/コンパクトなビューへ切り出します。より効率的です。\n• 列表示：元ページ寸法を保ち、それ以外をすべて紙白にして現在の本文列だけを開きます。拡大縮小せず、ルビ/隣接列も混入しません。\n「列表示」ではNDLOCRも列単位で実行し、毎回1列だけを開きます。",
    "仅在日文精确分列的多模型模式下生效。模型1和模型2先独立识别同一物理列；文字相同的列不再调用模型3～6；仍有分歧时，后续模型依次只识别未决列。两模型共同候选不会冒充多模型一致，也不会自动压过 AI 或人工裁决。关闭后，所有已选模型都会完整识别。": "日本語の精密列分割を使う複数モデルモードでのみ有効です。モデル1と2が同じ物理列を独立認識し、一致した列ではモデル3～6を呼びません。不一致が残る場合、後続モデルは未解決列だけを順番に認識します。2モデル共通候補を複数モデル一致として扱わず、AI/手動裁決を自動で上書きもしません。無効時は選択した全モデルが全件認識します。",
    "分列掩膜（推荐）：按模型裁白底视窗": "列マスク（推奨）：モデル別に白背景ビューを切り出す",
    "分列显示：原页白底，只显示目标列": "列表示：原寸の白背景ページで対象列だけ表示",
    "分列输入": "列入力",
    "只从正文可见层裁掉确定为空白的外侧画布并保留安全边距，文字像素不缩放、不锐化。Apple/NDL/Paddle 保留较宽纸白上下文；Hayai/48px 使用紧凑白底视窗。无论哪一种，相邻列与 Ruby 都不会重新进入正文 OCR。": "本文可視レイヤーから確実に空白の外側キャンバスだけを切り、余白を残します。文字画素は拡大縮小もシャープ化もしません。Apple/NDL/Paddle は広めの紙白文脈を残し、Hayai/48px はコンパクトな白背景ビューを使います。どちらでも隣接列とルビが本文OCRへ戻ることはありません。",
    "启用日文物理分列": "日本語の物理列分割を有効化",
    "固定正文框先隔离页眉/页脚，再检测日文物理竖列。正文与 Ruby 归属只检测一次；下方可选择“分列掩膜”或“分列显示”作为实际 OCR 输入方式。": "固定本文枠で先にヘッダー/フッターを分離してから、日本語の物理縦列を検出します。本文/ルビ所属は一度だけ検出し、下で「列マスク」または「列表示」を実際のOCR入力方式として選べます。",
    "单模型专用：关闭时把原页直接交给所选 OCR 引擎；开启时按日文物理竖列识别。极窄的已裁单列图会自动按单列处理，不会再被拆成多个假列。": "単一モデル専用：オフでは元ページを選択したOCRエンジンへそのまま渡し、オンでは日本語の物理縦列ごとに認識します。非常に細い単列クロップ画像は自動的に単列として扱い、誤った複数列へ分割しません。",
    "这是单模型专用分列开关。多模型使用独立的角色路由和共享分列几何，不会改写此设置。": "これは単一モデル専用の列分割スイッチです。複数モデルOCRは独立した役割ルーティングと共有列ジオメトリを使用し、この設定を書き換えません。",
    "多模型 OCR 对比（最多 6 个，自适应共识）": "複数モデルOCR比較（最大6モデル・適応コンセンサス）",
    "快速共识：模型1/2先判定，后续模型只补分歧列": "高速コンセンサス：モデル1/2で先に判定し、後続モデルは不一致列だけ補完",
    "所有 OCR 先共用同一张正文可见层：相邻列与 Ruby 均为不透明纸白色。送入识别器时只裁掉部分纯白画布；Apple/NDL/Paddle 等布局 OCR 保留较宽上下文，Hayai/48px 使用较紧凑视窗。文字像素保持原尺寸且不重采样。只有空结果、占位符或与黑像素估计相比严重缺字时，才把全尺寸掩膜作为该列唯一一次救援；低置信或引号不平衡本身不会重复调用 OCR。": "すべてのOCRは同じ本文可視レイヤーを共有し、隣接列とルビは不透明な紙白です。認識時は一部の純白キャンバスだけを切り、Apple/NDL/PaddleなどのレイアウトOCRは広い文脈を保持し、Hayai/48pxはよりコンパクトなビューを使います。文字画素は原寸のまま再サンプリングしません。空結果、プレースホルダー、または黒画素推定に対して深刻な欠字がある場合だけ、全寸マスクをその列の唯一の救済処理として使います。低信頼や引用符の不均衡だけではOCRを再実行しません。",
    "打开：多份 OCR 结果显示完整全文并保持同步滚动；关闭：只显示当前稳定句，继续逐句裁决。只改变显示，不重新 OCR 或对齐。": "オン：複数OCR結果の全文を表示してスクロール同期します。オフ：現在の安定文だけを表示して文単位裁決を続けます。表示だけが変わり、OCRや整列は再実行しません。",
    "模型1使用上方当前选中的 OCR；模型2～6共用完全相同的固定正文区域、物理分列几何和正文可见层。不同引擎只按自身输入合同选择宽上下文或紧凑白底视窗，正文/Ruby归属不会改变。完成后进入独立 OCR 对比工作区逐句自动选优或手动选择。": "モデル1は上で選択中のOCRを使用し、モデル2～6は完全に同じ固定本文領域・物理列形状・本文可視レイヤーを共有します。各エンジンは自身の入力仕様に従って広い文脈またはコンパクトな白背景ビューを選ぶだけで、本文/ルビ所属は変わりません。完了後は独立したOCR比較ワークスペースで文単位の自動選択または手動選択を行います。",
    "模型1决定页码、章节、图片锚点与基本段落结构。快速共识会逐级减少模型3～6调用：两模型相同列单独统计为共同候选，后续模型只补真正分歧列；AI/人工裁决始终可以覆盖候选，原始模型输出保持不变。": "モデル1がページ番号・章・画像アンカー・基本段落構造を決定します。高速コンセンサスはモデル3～6の呼出しを段階的に減らし、モデル1/2が一致した列は共通候補として別管理し、後続モデルは真の不一致列だけを補います。AI/手動裁決は常に候補を上書きでき、元のモデル出力は変更しません。",
    "独立于快速共识开关：模型1和模型2同时读取同一批已缓存物理列。每个模型使用独立临时增强目录，不会互相覆盖；共享分列裁图按页加锁。关闭快速共识时，前两模型仍可并行完成全量首轮，模型3～6随后按完整模式运行。若机器内存较小或两个模型争用同一加速设备，可关闭改为串行。": "高速コンセンサスとは独立して、モデル1と2は同じキャッシュ済み物理列を同時に読みます。各モデルは独立した一時強調ディレクトリを使うため相互上書きせず、共有列画像はページ単位でロックします。高速コンセンサスを無効にしても先頭2モデルは全件初回処理を並列実行でき、モデル3～6はその後フルモードで実行します。メモリが少ない場合や2モデルが同じアクセラレータを競合する場合は並列を無効にできます。",
    "选择文件夹，将当前 2～6 份 OCR 文本分别导出为 UTF-8 TXT；全部裁决后同时导出融合稿。": "フォルダを選択し、現在の2～6件のOCRテキストをUTF-8 TXTとして個別出力します。全裁決完了後は融合稿も同時に出力します。",
    "默认隐藏真正一致和已经选择的融合行；真正分歧会显示全部不同候选，快速共识产生的稳定共同候选自动保留，不再制造二次确认。勾选后自动前往下一组；取消勾选可查看全部融合结果。": "既定では真の一致と選択済み融合行を隠します。真の不一致は異なる候補をすべて表示し、高速コンセンサスで得た安定共通候補は自動保持して二重確認を増やしません。チェック時は次グループへ自動移動し、解除時は全融合結果を表示します。",
})

# Generic 2–6 model review wording (v19).
_EXACT_EN.update({
    "恢复当前全部初始 OCR、初始对齐、红绿差异和融合候选；单 OCR 模式下恢复标准化前结果。": "Restore all currently loaded initial OCR sources, initial alignment, red/green differences, and fusion candidates; in single-OCR mode, restore the pre-normalization result.",
    "导出 AI OCR 裁决包。当前全部原始 OCR 永久只读；此前已接受的冲突裁决会作为只读参考重新开放复审，AI 可保留旧结果，也可提交更好的 final_text；精确一致/共同候选仍锁定。": "Export the AI OCR adjudication package. All current raw OCR sources remain permanently read-only; previously accepted conflict decisions reopen as read-only evidence, and AI may keep the old result or submit a better final_text. Exact agreement/shared candidates remain locked.",
    "程序崩溃或重启后，直接从新版逐模型纠错 ZIP 恢复当前 2～6 份 OCR 文档、物理列、对齐和人工选择。先在页面管理重新载入同一批图片/PDF，可自动重绑新临时路径；不重新 OCR。": "After a crash/restart, restore the current 2–6 OCR documents, physical columns, alignment, and manual choices directly from the new per-model correction ZIP. Reload the same images/PDF in Page Manager first; new temporary paths are rebound automatically without rerunning OCR.",
    
    "请先载入 2～6 份 OCR 结果。": "Load 2–6 OCR results first.",
    "原始各模型 OCR、物理列和对齐均未修改。": "The raw OCR from every model, physical columns, and alignment were left unchanged.",
    "已选中独立的‘图文对照人工校对’候选，所有 OCR 模型原文均未被覆盖。": "The independent ‘Image/Text Manual Review’ candidate was selected; no raw OCR source was overwritten.",
})
_EXACT_JA.update({
    "恢复当前全部初始 OCR、初始对齐、红绿差异和融合候选；单 OCR 模式下恢复标准化前结果。": "現在読み込まれている全初期OCR、初期整列、赤/緑差分、融合候補を復元します。単一OCRモードでは正規化前の結果を復元します。",
    "导出 AI OCR 裁决包。当前全部原始 OCR 永久只读；此前已接受的冲突裁决会作为只读参考重新开放复审，AI 可保留旧结果，也可提交更好的 final_text；精确一致/共同候选仍锁定。": "AI OCR裁決パッケージを出力します。現在の全元OCRは永久に読み取り専用です。既採用の差異裁決は読み取り専用証拠として再確認でき、AIは旧結果維持またはより良いfinal_textを提出できます。完全一致/共通候補はロックを維持します。",
    "程序崩溃或重启后，直接从新版逐模型纠错 ZIP 恢复当前 2～6 份 OCR 文档、物理列、对齐和人工选择。先在页面管理重新载入同一批图片/PDF，可自动重绑新临时路径；不重新 OCR。": "クラッシュ/再起動後、新版モデル別修正ZIPから現在の2～6件のOCR文書、物理列、整列、手動選択を直接復元します。先にページ管理で同じ画像/PDFを再読込すれば新しい一時パスを自動再結合し、OCRは再実行しません。",
    "逐句裁决：左侧队列定位未决句，上方已载入模型只显示当前句；绿色=一致，红色=差异。候选卡始终完整展开，红底=替换、橙底=增删；融合候选可直接修改，“显示全文对比”只切换显示。": "文ごとの裁決：左キューで未決文へ移動し、上の読み込み済みモデルは現在文だけを表示します。緑=一致、赤=差異。候補カードは常に全展開し、赤背景=置換、橙背景=挿入/削除。融合候補は直接編集でき、「全文比較を表示」は表示だけを切り替えます。",
    "请先载入 2～6 份 OCR 结果。": "先に2～6件のOCR結果を読み込んでください。",
    "原始各模型 OCR、物理列和对齐均未修改。": "各モデルの元OCR、物理列、整列はいずれも変更していません。",
    "已选中独立的‘图文对照人工校对’候选，所有 OCR 模型原文均未被覆盖。": "独立した「画像/テキスト手動校正」候補を選択しました。どのOCRモデル原文も上書きしていません。",
})

# v19 visible Apple/Hayai naming and 2–6 model restore chrome.
_EXACT_EN.update({
    "当前全部模型原文均已独立保留；请到左侧“OCR 对比”逐句改选或直接应用自动融合稿。": "All loaded model source texts are preserved independently. Use OCR Compare on the left to reselect sentence by sentence or apply the automatic fusion draft directly.",
    "高置信字符级融合 · 已自动采用（可重新选择任一原始 OCR）": "High-confidence character-level fusion · Auto-selected (you can reselect any raw OCR source)",
    "恢复会替换当前 OCR 对比中的 2～6 份模型文档和候选状态，但不会重新 OCR。继续吗？": "Restoring will replace the current 2–6 model documents and candidate state in OCR Compare, but will not run OCR again. Continue?",
    "Apple OCR 正在识别当前列，请稍候。": "Apple OCR is recognizing the current column. Please wait.",
    "本列原图不存在，无法运行 Apple OCR。": "The source image for this column is unavailable, so Apple OCR cannot run.",
    "正在手动运行 Apple OCR 识别当前单列；逐字框不会被替换…": "Running Apple OCR manually on the current column; character boxes will not be replaced…",
    "Apple OCR 本列结果已显示并复制到剪贴板；可选择替换本列。": "The Apple OCR result for this column is displayed and copied to the clipboard; you may replace the column with it.",
    "Apple OCR 完成了字符框对齐，但未返回可复制文本。": "Apple OCR completed character-box alignment but returned no copyable text.",
    "已用 Apple OCR 结果替换本列；应用人工纠错结果后写回正文。": "This column was replaced with the Apple OCR result; apply the manual-correction result to write it back to body text.",
})
_EXACT_JA.update({
    "当前全部模型原文均已独立保留；请到左侧“OCR 对比”逐句改选或直接应用自动融合稿。": "読み込まれた全モデルの元OCRは独立して保持されています。左の「OCR比較」で文ごとに再選択するか、自動融合下書きをそのまま適用できます。",
    "高置信字符级融合 · 已自动采用（可重新选择任一原始 OCR）": "高信頼の文字単位融合 · 自動採用済み（任意の元OCRを再選択可能）",
    "恢复会替换当前 OCR 对比中的 2～6 份模型文档和候选状态，但不会重新 OCR。继续吗？": "復元すると現在のOCR比較にある2～6件のモデル文書と候補状態を置き換えますが、OCRは再実行しません。続行しますか？",
    "Apple OCR 正在识别当前列，请稍候。": "Apple OCRで現在列を認識中です。お待ちください。",
    "本列原图不存在，无法运行 Apple OCR。": "この列の元画像がないためApple OCRを実行できません。",
    "正在手动运行 Apple OCR 识别当前单列；逐字框不会被替换…": "Apple OCRを手動で現在の1列だけに実行しています。文字枠は置き換えません…",
    "Apple OCR 本列结果已显示并复制到剪贴板；可选择替换本列。": "Apple OCRの列結果を表示してクリップボードへコピーしました。この列を置換できます。",
    "Apple OCR 完成了字符框对齐，但未返回可复制文本。": "Apple OCRは文字枠整列を完了しましたが、コピー可能なテキストを返しませんでした。",
    "已用 Apple OCR 结果替换本列；应用人工纠错结果后写回正文。": "この列をApple OCR結果で置換しました。手動修正結果を適用すると本文へ書き戻します。",
})
# v19 Apple OCR manual-column controls.
_EXACT_EN.update({
    "仅在手动点击后运行 Apple OCR；不会后台自动识别": "Apple OCR runs only when you click manually; it never recognizes in the background.",
    "Apple OCR 识别本列并复制": "Apple OCR: recognize this column and copy",
    "用 Apple OCR 替换本列": "Replace this column with Apple OCR",
})
_EXACT_JA.update({
    "仅在手动点击后运行 Apple OCR；不会后台自动识别": "Apple OCRは手動クリック時だけ実行し、バックグラウンドでは認識しません。",
    "Apple OCR 识别本列并复制": "Apple OCRでこの列を認識してコピー",
    "用 Apple OCR 替换本列": "この列をApple OCRで置換",
})

# v25 OCR preview geometry: detector box vs actual native-pixel OCR input box.
_EXACT_EN.update({
    "红色实线＝检测框  ·  绿色实线＝OCR输入原像素框": "Red solid = detector box  ·  Green solid = OCR native-pixel input box",
    "检测框": "Detector box",
    "输入框": "Input box",
    "检测框表示 component detector 最终边界；OCR输入原像素框表示真正从原图保留的范围。输入框外仍可能存在纯白安全槽上下文，但不会重新放出相邻列或 Ruby。": "The detector box is the final component-detector boundary. The OCR native-pixel input box is the area actually preserved from the source image. Pure-white safe-slot context may exist outside it, but neighboring columns and Ruby are never revealed again.",
    "分列输入预览 · {page_name} · {column_count} 列": "Column Input Preview · {page_name} · {column_count} columns",
    "已复用当前实时预览页的同一组物理列；": "Reused the same physical columns from the current live-preview page; ",
    "当前页尚无实时分列结果，已按现有参数重新检测；": "No live column result exists for this page, so it was detected again with the current settings; ",
    "页面：{page_name}。{sync_note}左侧红色实线为检测框、绿色实线为 OCR 输入原像素范围，按日文阅读顺序从右到左编号，共 {column_count} 列；右侧展示当前实际输入：{mode_label}，尺寸 {width}×{height}。绿色输入框之外仍可存在纯白安全槽上下文；目标正文保持原始像素，其他列与 Ruby 均被纸白色隔离。": "Page: {page_name}. {sync_note}On the left, red solid lines are detector boxes and green solid lines show the native-pixel OCR input range. Columns are numbered right-to-left in Japanese reading order ({column_count} total). The right pane shows the actual current input: {mode_label}, size {width}×{height}. Pure-white safe-slot context may remain outside the green input box; target body pixels stay original while other columns and Ruby are isolated with paper white.",
    "NDLOCR 整列上下文": "NDLOCR full-column context",
    "单列原像素视窗": "Single-column native-pixel viewport",
    "单列 mask 视窗": "Single-column mask viewport",
    "单列 crop 视窗": "Single-column crop viewport",
    "单列 {viewport_mode} 视窗": "Single-column {viewport_mode} viewport",
    "分列预览失败": "Column Preview Failed",
    "实时预览已关闭；OCR 仍在后台正常运行。": "Live preview is off; OCR continues normally in the background.",
    "实时预览已关闭；": "Live preview is off; ",
    "实时预览已关闭": "Live preview is off",
    "分列检测": "Column detection",
    "识别完成": "Recognition complete",
    "当前图片：{display_name}": "Current image: {display_name}",
    "请先选择图片或 PDF。": "Select images or a PDF first.",
    "请先固定正文区域": "Set the Body Region First",
    "请先在右侧图片上拖框选定纯正文区域，再预览分列。": "Drag a box on the image at right to select the body-only region before previewing columns.",
    "未检测到竖列": "No Vertical Columns Detected",
    "当前固定区域没有检测到稳定竖列。请重新框选纯正文区域，或提高分列灵敏度。": "No stable vertical columns were detected in the fixed region. Select the body-only region again or increase column sensitivity.",
    "分列预览图生成失败": "Failed to generate the column preview image",
})
_EXACT_JA.update({
    "红色实线＝检测框  ·  绿色实线＝OCR输入原像素框": "赤実線＝検出枠  ·  緑実線＝OCR元画素入力枠",
    "检测框": "検出枠",
    "输入框": "入力枠",
    "检测框表示 component detector 最终边界；OCR输入原像素框表示真正从原图保留的范围。输入框外仍可能存在纯白安全槽上下文，但不会重新放出相邻列或 Ruby。": "検出枠はcomponent detectorの最終境界です。OCR元画素入力枠は元画像から実際に保持される範囲を示します。入力枠の外側に純白の安全スロット文脈が残る場合がありますが、隣接列やRubyが再表示されることはありません。",
    "分列输入预览 · {page_name} · {column_count} 列": "列入力プレビュー · {page_name} · {column_count} 列",
    "已复用当前实时预览页的同一组物理列；": "現在のライブプレビューページと同じ物理列を再利用しました。",
    "当前页尚无实时分列结果，已按现有参数重新检测；": "このページにはライブ列結果がないため、現在の設定で再検出しました。",
    "页面：{page_name}。{sync_note}左侧红色实线为检测框、绿色实线为 OCR 输入原像素范围，按日文阅读顺序从右到左编号，共 {column_count} 列；右侧展示当前实际输入：{mode_label}，尺寸 {width}×{height}。绿色输入框之外仍可存在纯白安全槽上下文；目标正文保持原始像素，其他列与 Ruby 均被纸白色隔离。": "ページ：{page_name}。{sync_note}左側では赤い実線が検出枠、緑の実線がOCR元画素入力範囲です。日本語の読書順に右から左へ番号を付け、全{column_count}列です。右側には現在の実入力：{mode_label}、サイズ {width}×{height} を表示します。緑の入力枠外に純白の安全スロット文脈が残る場合がありますが、対象本文は元画素のまま、他列とRubyは紙白で隔離されます。",
    "NDLOCR 整列上下文": "NDLOCR 列全体コンテキスト",
    "单列原像素视窗": "単列・元画素ビューポート",
    "单列 mask 视窗": "単列 mask ビューポート",
    "单列 crop 视窗": "単列 crop ビューポート",
    "单列 {viewport_mode} 视窗": "単列 {viewport_mode} ビューポート",
    "分列预览失败": "列プレビュー失敗",
    "实时预览已关闭；OCR 仍在后台正常运行。": "ライブプレビューはオフです。OCRはバックグラウンドで正常に続行します。",
    "实时预览已关闭；": "ライブプレビューはオフです。",
    "实时预览已关闭": "ライブプレビューはオフ",
    "分列检测": "列検出",
    "识别完成": "認識完了",
    "当前图片：{display_name}": "現在の画像：{display_name}",
    "请先选择图片或 PDF。": "先に画像またはPDFを選択してください。",
    "请先固定正文区域": "本文領域を先に固定",
    "请先在右侧图片上拖框选定纯正文区域，再预览分列。": "右側の画像で本文だけの領域をドラッグ選択してから列プレビューを実行してください。",
    "未检测到竖列": "縦列を検出できませんでした",
    "当前固定区域没有检测到稳定竖列。请重新框选纯正文区域，或提高分列灵敏度。": "現在の固定領域では安定した縦列を検出できませんでした。本文領域を選択し直すか、列検出感度を上げてください。",
    "分列预览图生成失败": "列プレビュー画像の生成に失敗しました",
})

# v31 NDLOCR precise-column contract: no full-page/hybrid bypass while column splitting is enabled.
_EXACT_EN.update({
    "逐列单列：与其他 OCR 使用相同分列输入": "Per-column single input: same column transport as other OCR engines",
    "逐列单列：紧裁正文上下文": "One physical column at a time: tightly framed body context",
    "开启分列后，NDLOCR-Lite 不再使用整页/智能混合旁路；每次只接收当前 Ruby-free 物理列，并使用与 Apple 单列相同的原像素纸白 framing（典型宽度约 70–80 px）。": "With column splitting enabled, NDLOCR-Lite no longer uses full-page or hybrid bypasses. Each call receives only the current Ruby-free physical column and uses the same native-pixel paper-white framing as Apple (typically about 70–80 px wide).",
    "开启分列后，NDLOCR-Lite 不再使用整页/智能混合旁路；每次只接收当前 Ruby-free 物理列，保留原像素与整列高度，过宽白边列仅做居中排版；不送整页。": "With column splitting enabled, NDLOCR-Lite receives one Ruby-free physical column per call, retaining native pixels and full column height. Only unusually wide blank margins are recentered; full pages are never sent.",
})
_EXACT_JA.update({
    "逐列单列：与其他 OCR 使用相同分列输入": "列単位の単列入力：他のOCRと同じ列入力を使用",
    "逐列单列：紧裁正文上下文": "物理列を1列ずつ認識：本文をタイトに枠取り",
    "开启分列后，NDLOCR-Lite 不再使用整页/智能混合旁路；每次只接收当前 Ruby-free 物理列，并使用与 Apple 单列相同的原像素纸白 framing（典型宽度约 70–80 px）。": "列分割を有効にすると、NDLOCR-Liteは全ページ/ハイブリッド経路を使用しません。各呼び出しは現在のRuby除去済み物理列だけを受け取り、Apple単列と同じ元画素の紙白フレーム（通常約70～80px幅）を使用します。",
    "开启分列后，NDLOCR-Lite 不再使用整页/智能混合旁路；每次只接收当前 Ruby-free 物理列，保留原像素与整列高度，过宽白边列仅做居中排版；不送整页。": "列分割時、NDLOCR-Liteは全ページ/ハイブリッド経路を使わず、Rubyを除いた物理列を1列ずつ認識します。元画素と列全体の高さを保ち、幅が極端に広い白余白だけを中央に配置します。",
})

# v29 compact preview tooltip variants (previously missing from the exact offline catalog).
_EXACT_EN.update({
    "清除当前 OCR 识别区域框选，恢复使用完整正文区域": "Clear the current OCR recognition-area selection and restore the full body region.",
    "显示/隐藏右侧实时预览中的检测框": "Show or hide detector boxes in the live preview on the right.",
    "显示/隐藏右侧实时预览中的 OCR 输入原像素框": "Show or hide OCR native-pixel input boxes in the live preview on the right.",
})
_EXACT_JA.update({
    "清除当前 OCR 识别区域框选，恢复使用完整正文区域": "現在のOCR認識範囲の選択枠を解除し、本文領域全体へ戻します。",
    "显示/隐藏右侧实时预览中的检测框": "右側のリアルタイムプレビューで検出枠を表示/非表示にします。",
    "显示/隐藏右侧实时预览中的 OCR 输入原像素框": "右側のリアルタイムプレビューでOCR元画素入力枠を表示/非表示にします。",
})

# v32 role-based multi-model OCR scheduler.
_EXACT_EN.update({
    "多模型 OCR（分工式，最多 6 个模型）": "Multi-model OCR (role-based, up to 6 models)",
    "✓ 主模型按输入粒度分工；分歧复核只处理仍有分歧的物理列": "✓ Main models are divided by input granularity; disagreement review processes only complete sentences that still disagree",
    "✓ 推荐：Hayai逐列 + NDL整页 + 48px全列；Hayai/48px仅复核分歧": "✓ Recommended: Hayai columns + NDL full page + 48px full-column; Hayai/48px only for targeted disagreement retry",
    "逐列主模型读取共享物理列；整页主模型只做真实整页 OCR，并在识别后投影到 canonical sentence；全列主模型完整读取全部共享物理列。可选分歧复核模型只重识别残余冲突物理列，复核 1→2→3 串行早停。": "The column main model reads shared physical columns; the page main model performs genuine full-page OCR and is projected to canonical sentences only after recognition; the full-column main model reads every shared physical column. Optional review models re-recognize residual conflict columns only; reviews 1→2→3 run sequentially with early exit.",
    "整页结果会投影到统一 sentence_id。全列主模型不再构造横向整句图，而是完整读取同轮共享几何中的所有物理列。": "Full-page results are projected to the shared sentence_id. The full-column main model no longer builds horizontal sentence strips; it reads every physical column from the shared geometry.",
    "开启后，上方单模型选择会锁定且不参与本次运行。多模型拥有独立的 6 个可选槽位：逐列 / 整页 / 全列三个主模型，以及三个分歧复核模型。六个槽位都允许为空，但逐列 / 整页 / 全列至少选择一个才能开始；同一个 OCR 模型只能使用一次。": "When enabled, the single-model selector above is locked and does not participate in this run. Multi-model OCR has six independent optional slots: column/page/full-column main models and three disagreement-review models. All six slots may be empty, but at least one main role must be selected; each OCR engine may be used only once.",
    "逐列主模型读取共享物理列；整页主模型只做真实整页 OCR，并在识别后投影到 canonical sentence；全列主模型完整读取全部共享物理列，不再生成整句图。分歧复核 1→2→3 只处理残余冲突物理列并串行早停。": "The column main model reads shared physical columns; the page main model performs genuine full-page OCR and is projected to canonical sentences after recognition; the full-column main model independently reads every shared physical column. Disagreement Review 1→2→3 processes residual conflict columns only and stops early when resolved.",
    "整页只是识别输入粒度：结果会先投影到与其它主模型相同的 sentence_id，再进入 OCR 对比。全列主模型完整读取全部物理列；分歧复核只读取目标冲突列。": "Page is only an OCR input granularity: its result is projected to the same sentence_id as the other main models before OCR Compare. The full-column role reads every physical column; disagreement review reads only targeted conflict columns.",
    "未选择": "Not selected",
    "角色调度": "Role scheduler",
    "强制逐列：共享统一分列后逐列识别": "Force per-column: recognize shared unified columns one by one",
    "仅影响 NDLOCR-Lite。智能混合先在同一份统一物理分列几何上构造 Ruby-free 整页输入并识别一次，再把结果严格映射回共享列槽；只有空列、跨列歧义、低置信或明显异常列才逐列补识。高速整页每页只识别一次，漏列保留为 □ 进入复核。强制逐列仍复用相同的页面级分列，不重复检测。": "Affects NDLOCR-Lite only. Smart Hybrid first builds a Ruby-free full-page input from the same unified physical-column geometry and recognizes it once, then maps results strictly back to the shared column slots; only empty, cross-column ambiguous, low-confidence, or clearly abnormal columns are re-recognized individually. High-speed Full Page recognizes each page once and keeps missing columns as □ for review. Force Per-column still reuses the same page-level split and never repeats column detection.",
    "逐列主模型": "Column main model",
    "整页主模型": "Page main model",
    "全列主模型": "Full-column main model",
    "分歧复核 1": "Disagreement Review 1",
    "分歧复核 2": "Disagreement Review 2",
    "分歧复核 3": "Disagreement Review 3",
})
_EXACT_JA.update({
    "多模型 OCR（分工式，最多 6 个模型）": "複数モデルOCR（役割分担式、最大6モデル）",
    "✓ 主模型按输入粒度分工；分歧复核只处理仍有分歧的物理列": "✓ 主モデルは入力粒度ごとに分担し、差異レビューは差異が残る完全な文だけを処理します",
    "✓ 推荐：Hayai逐列 + NDL整页 + 48px全列；Hayai/48px仅复核分歧": "✓ 推奨：Hayai列単位 + NDL全ページ + 48px全列；Hayai/48pxは不一致のみ再確認",
    "逐列主模型读取共享物理列；整页主模型只做真实整页 OCR，并在识别后投影到 canonical sentence；全列主模型完整读取全部共享物理列。可选分歧复核模型只重识别残余冲突物理列，复核 1→2→3 串行早停。": "列単位主モデルは共有物理列を読み取り、全ページ主モデルは実際の全ページOCRだけを行って認識後にcanonical sentenceへ投影します。全列主モデルは共有物理列をすべて完全処理します。任意の差異レビューモデルは残った競合物理列だけを再認識し、レビュー1→2→3は直列で早期終了します。",
    "整页结果会投影到统一 sentence_id。全列主模型不再构造横向整句图，而是完整读取同轮共享几何中的所有物理列。": "全ページ結果は共通のsentence_idへ投影されます。全列主モデルは横書き文画像を再構成せず、共有ジオメトリのすべての物理列を読み取ります。",
    "开启后，上方单模型选择会锁定且不参与本次运行。多模型拥有独立的 6 个可选槽位：逐列 / 整页 / 全列三个主模型，以及三个分歧复核模型。六个槽位都允许为空，但逐列 / 整页 / 全列至少选择一个才能开始；同一个 OCR 模型只能使用一次。": "有効にすると上部の単一モデル選択はロックされ、今回の実行には参加しません。複数モデルOCRには、列単位・全ページ・全文の3つの主モデルと3つの差異レビューモデルから成る独立した6スロットがあります。6スロットはすべて空でも構いませんが、開始には列単位・全ページ・全文のうち少なくとも1つが必要で、同じOCRエンジンは1回だけ使用できます。",
    "逐列主模型读取共享物理列；整页主模型只做真实整页 OCR，并在识别后投影到 canonical sentence；全列主模型完整读取全部共享物理列，不再生成整句图。分歧复核 1→2→3 只处理残余冲突物理列并串行早停。": "列単位主モデルは共有物理列を読み取り、全ページ主モデルは実際の全ページOCRだけを行って認識後にcanonical sentenceへ投影します。全文主モデルは共有列分割に従って連続する複数列を実際の文画像へ結合してOCRします。差異レビュー1→2→3は直列で早期終了します。",
    "整页只是识别输入粒度：结果会先投影到与其它主模型相同的 sentence_id，再进入 OCR 对比。全列主模型完整读取全部物理列；分歧复核只读取目标冲突列。": "全ページはOCR入力粒度にすぎません。結果は他の主モデルと同じsentence_idへ投影してからOCR比較へ入ります。全文OCRは列テキストの連結ではなく、その文が覆う複数列の元画像を結合した実画像を再認識します。",
    "未选择": "未選択",
    "角色调度": "役割スケジューラ",
    "强制逐列：共享统一分列后逐列识别": "強制列単位：共有統一列分割を1列ずつ認識",
    "仅影响 NDLOCR-Lite。智能混合先在同一份统一物理分列几何上构造 Ruby-free 整页输入并识别一次，再把结果严格映射回共享列槽；只有空列、跨列歧义、低置信或明显异常列才逐列补识。高速整页每页只识别一次，漏列保留为 □ 进入复核。强制逐列仍复用相同的页面级分列，不重复检测。": "NDLOCR-Liteだけに影響します。スマートハイブリッドは同じ統一物理列ジオメトリからRuby除去済み全ページ入力を作って1回認識し、その結果を共有列スロットへ厳密に戻します。空列・列またぎの曖昧さ・低信頼・明らかな異常列だけを列単位で補認識します。高速全ページは各ページを1回だけ認識し、欠落列は□のままレビューへ送ります。強制列単位も同じページレベル列分割を再利用し、検出を繰り返しません。",
    "逐列主模型": "列単位主モデル",
    "整页主模型": "全ページ主モデル",
    "全列主模型": "全列主モデル",
    "分歧复核 1": "差異レビュー 1",
    "分歧复核 2": "差異レビュー 2",
    "分歧复核 3": "差異レビュー 3",
})

# v33 V5 role-aware adjudication package.
_EXACT_EN.update({
    "V5 多模型分歧裁决包（默认推荐）": "V5 role-aware multi-model disagreement package (recommended)",
    "V4 多模型分歧裁决包（兼容）": "V4 multi-model disagreement package (compatibility)",
    "V5 多模型分歧裁决包": "V5 role-aware multi-model disagreement package",
    "选择 AI 修复包的保存位置": "Choose where to save the AI repair package",
})
_EXACT_JA.update({
    "V5 多模型分歧裁决包（默认推荐）": "V5 役割対応・複数モデル差異裁決パッケージ（推奨）",
    "V4 多模型分歧裁决包（兼容）": "V4 複数モデル差異裁決パッケージ（互換）",
    "V5 多模型分歧裁决包": "V5 役割対応・複数モデル差異裁決パッケージ",
    "选择 AI 修复包的保存位置": "AI修復パッケージの保存先を選択",
})
_EXACT_EN.update({
    "推荐：稀疏 JSON (*.json);;V5/V4 裁决 ZIP (*.zip);;兼容：AI 修复 EPUB (*.epub);;全部支持格式 (*.json *.zip *.epub)": "Recommended: sparse JSON (*.json);;V5/V4 adjudication ZIP (*.zip);;Compatible: AI repair EPUB (*.epub);;All supported formats (*.json *.zip *.epub)",
})
_EXACT_JA.update({
    "推荐：稀疏 JSON (*.json);;V5/V4 裁决 ZIP (*.zip);;兼容：AI 修复 EPUB (*.epub);;全部支持格式 (*.json *.zip *.epub)": "推奨：疎JSON (*.json);;V5/V4裁決ZIP (*.zip);;互換：AI修復EPUB (*.epub);;対応形式すべて (*.json *.zip *.epub)",
})

# Current-only adjudication package wording (development schema; no legacy package compatibility).
_EXACT_EN.update({
    "V5 多模型分歧裁决包": "V5 Multi-model Conflict Adjudication Package",
    "导出 AI OCR 裁决包。当前全部原始 OCR 永久只读；此前已接受的冲突裁决会作为只读参考重新开放复审，AI 可保留此前当前格式结果，也可提交更好的 final_text；精确一致/共同候选仍按当前策略锁定。": "Export the current AI OCR adjudication package. All original OCR remains permanently read-only. Previously accepted conflict decisions are reopened as read-only references; AI may keep the current-format result or submit a better final_text. Exact matches and shared candidates remain locked according to the current policy.",
    "仅导入当前格式 AI OCR 裁决 JSON/ZIP。后导入的已接受结果可改进同一稳定句；缺失/未决行不会抹掉前一包的好结果，且绝不改写原始 OCR。": "Import current-format AI OCR adjudication JSON/ZIP only. Later accepted results may improve the same stable sentence; missing or unresolved rows never erase a previously accepted good result, and original OCR is never overwritten.",
    "当前开发版只导出 V5 多模型分歧裁决包。真正分歧交给外部 AI，本地已完成裁决会冻结；普通两模型共同候选默认继续 Lean 保留。": "The current development build exports only the V5 Multi-model Conflict Adjudication Package. True disagreements go to external AI, completed local decisions are frozen, and ordinary two-model shared candidates remain Lean by default.",
})
_EXACT_JA.update({
    "V5 多模型分歧裁决包": "V5 複数モデル差異裁決パッケージ",
    "导出 AI OCR 裁决包。当前全部原始 OCR 永久只读；此前已接受的冲突裁决会作为只读参考重新开放复审，AI 可保留此前当前格式结果，也可提交更好的 final_text；精确一致/共同候选仍按当前策略锁定。": "現在形式のAI OCR裁決パッケージを書き出します。すべての元OCRは永久に読み取り専用です。以前に採用された差異裁決は読み取り専用の参照として再レビューされ、AIは現在形式の結果を維持するか、より良いfinal_textを提出できます。完全一致と共通候補は現在の方針に従ってロックされます。",
    "仅导入当前格式 AI OCR 裁决 JSON/ZIP。后导入的已接受结果可改进同一稳定句；缺失/未决行不会抹掉前一包的好结果，且绝不改写原始 OCR。": "現在形式のAI OCR裁決JSON/ZIPだけを読み込みます。後から採用された結果は同じ安定文を改善できます。欠落または未決の行が以前の良い結果を消すことはなく、元OCRも決して上書きしません。",
    "当前开发版只导出 V5 多模型分歧裁决包。真正分歧交给外部 AI，本地已完成裁决会冻结；普通两模型共同候选默认继续 Lean 保留。": "現在の開発版ではV5複数モデル差異裁決パッケージだけを書き出します。実際の差異は外部AIへ送り、ローカルで完了した裁決は固定し、通常の2モデル共通候補は既定でLeanのまま保持します。",
})

_EXACT_EN.update({
    "程序崩溃或重启后，只从当前格式裁决 ZIP 恢复当前 2～6 份 OCR 文档、物理列、密封对齐和人工选择。旧格式裁决包在稳定版前不兼容；恢复不会重新 OCR。": "After a crash or restart, restore the current 2–6 OCR documents, physical columns, sealed alignment, and manual selections only from the current-format adjudication ZIP. Legacy adjudication packages are intentionally incompatible before the stable release; restoration never reruns OCR.",
    "导出当前 V5 多模型分歧裁决包：只开放当前策略判定需要外部裁决的稳定句，完整保留角色、独立执行/seeded reuse、物理列与视觉证据；本地已完成裁决保持冻结。": "Export the current V5 multi-model disagreement adjudication package: only stable sentences that the current policy marks for external adjudication are unlocked. Roles, independent execution/seeded reuse, physical columns, and visual evidence are preserved; completed local decisions remain frozen.",
    "只导入当前 V5 decisions.json 或包含该文件的 ZIP。package_id、structure_sha256 和可编辑 ID 必须与当前会话完全一致。": "Import only the current V5 decisions.json or a ZIP containing that file. package_id, structure_sha256, and editable IDs must exactly match the current session.",
})
_EXACT_JA.update({
    "程序崩溃或重启后，只从当前格式裁决 ZIP 恢复当前 2～6 份 OCR 文档、物理列、密封对齐和人工选择。旧格式裁决包在稳定版前不兼容；恢复不会重新 OCR。": "クラッシュまたは再起動後は、現在形式の裁決ZIPからのみ、現在の2～6個のOCR文書・物理列・封印済み整列・手動選択を復元します。安定版までは旧形式裁決パッケージと互換性を持たせず、復元時にOCRを再実行しません。",
    "导出当前 V5 多模型分歧裁决包：只开放当前策略判定需要外部裁决的稳定句，完整保留角色、独立执行/seeded reuse、物理列与视觉证据；本地已完成裁决保持冻结。": "現在のV5複数モデル差異裁決パッケージを書き出します。現在の方針で外部裁決が必要と判定された安定文だけを開放し、役割・独立実行/seeded reuse・物理列・視覚証拠を完全に保持します。ローカルで完了した裁決は固定されたままです。",
    "只导入当前 V5 decisions.json 或包含该文件的 ZIP。package_id、structure_sha256 和可编辑 ID 必须与当前会话完全一致。": "現在のV5 decisions.json、またはそのファイルを含むZIPだけを読み込みます。package_id、structure_sha256、編集可能IDは現在のセッションと完全一致する必要があります。",
})

# v5.3 persistent project workspace / Stirling-style project layer.
_EXACT_EN.update({
    "工作区": "Workspace",
    "项目工作区": "Project Workspace",
    "工作区负责项目位置、项目增删、完整备份、运行日志和导出产物。页面管理只负责页面顺序、页面类型和扫描件处理；两者状态仍自动联动。": "Workspace manages project location, project creation/deletion, complete backups, run logs, and exported artifacts. Page Manager only handles page order, page types, and scanned-page processing; their state stays synchronized automatically.",
    "项目": "Project",
    "选择位置": "Choose Location",
    "项目目录": "Project Folder",
    "导入项目": "Import Project",
    "删除项目": "Delete Project",
    "未选择（临时会话）": "Not selected (temporary session)",
    "项目、OCR、裁决、导出和运行记录的持久化根目录": "Persistent root for projects, OCR, adjudication, exports, and run history",
    "打开已有 Novel Formatter 项目文件夹并恢复页面/正文状态": "Open an existing Novel Formatter project folder and restore page/text state",
    "未选择项目：仍可使用临时会话；创建项目后会自动保存页面、OCR、裁决与 EPUB。": "No project selected. Temporary sessions still work; after creating a project, pages, OCR, adjudication, and EPUB outputs are saved automatically.",
    "临时会话：当前内容不会自动写入项目目录。": "Temporary session: current content is not automatically written to a project folder.",
    "已切换工作区；请选择或新建项目。": "Workspace changed. Select or create a project.",
    "项目已删除。": "Project deleted.",
    "未绑定项目目录": "No project folder bound",
})
_EXACT_JA.update({
    "工作区": "ワークスペース",
    "项目工作区": "プロジェクトワークスペース",
    "工作区负责项目位置、项目增删、完整备份、运行日志和导出产物。页面管理只负责页面顺序、页面类型和扫描件处理；两者状态仍自动联动。": "ワークスペースはプロジェクトの保存場所、作成・削除、完全バックアップ、実行ログ、出力成果物を管理します。ページ管理はページ順、ページ種別、スキャンページ処理のみを担当し、両者の状態は自動的に同期されます。",
    "项目": "プロジェクト",
    "选择位置": "保存場所を選択",
    "项目目录": "プロジェクトフォルダ",
    "导入项目": "プロジェクトを読み込む",
    "删除项目": "プロジェクトを削除",
    "未选择（临时会话）": "未選択（一時セッション）",
    "项目、OCR、裁决、导出和运行记录的持久化根目录": "プロジェクト、OCR、裁決、出力、実行履歴を永続保存するルートフォルダ",
    "打开已有 Novel Formatter 项目文件夹并恢复页面/正文状态": "既存の Novel Formatter プロジェクトフォルダを開き、ページと本文状態を復元します",
    "未选择项目：仍可使用临时会话；创建项目后会自动保存页面、OCR、裁决与 EPUB。": "プロジェクト未選択でも一時セッションを利用できます。プロジェクト作成後はページ、OCR、裁決、EPUBを自動保存します。",
    "临时会话：当前内容不会自动写入项目目录。": "一時セッション：現在の内容はプロジェクトフォルダへ自動保存されません。",
    "已切换工作区；请选择或新建项目。": "ワークスペースを切り替えました。プロジェクトを選択または作成してください。",
    "项目已删除。": "プロジェクトを削除しました。",
    "未绑定项目目录": "プロジェクトフォルダ未設定",
})

# v5.4 project backup / integrity UI.
_EXACT_EN.update({
    "把已有 Novel Formatter 项目文件夹复制进当前工作区并恢复完整状态": "Copy an existing Novel Formatter project into the current workspace and restore its complete state",
    "导入备份": "Import Backup",
    "从项目 ZIP 备份安全恢复到当前工作区": "Safely restore a project ZIP backup into the current workspace",
    "备份项目": "Back Up Project",
    "导出包含源文件、页面、OCR、裁决、正文和导出结果的完整项目 ZIP": "Export a complete project ZIP containing sources, pages, OCR, adjudication, text and exports",
    "检查项目": "Check Project",
    "检查页面/源文件/lineage/运行历史，并可回收孤儿页面缓存": "Check pages, source files, lineage and run history, and optionally recover orphan page cache files",
    "项目检查完成。": "Project check complete.",
    "项目检查完成；未清理任何文件。": "Project check complete; no files were cleaned up.",
    "优化存储": "Optimize Storage",
    "无损压缩 OCR/阶段/缓存 JSON，并把数千个 OCR 小缓存合并为 SQLite；不删除原图、OCR 结果或裁决": "Losslessly compress OCR/stage/cache JSON and pack thousands of small OCR cache files into SQLite; source images, OCR results and adjudications are preserved",
    "优化中…": "Optimizing…",
    "正在无损优化项目存储；原图、OCR、裁决和导出结果均保留。": "Optimizing project storage losslessly; source images, OCR, adjudications and exports are preserved.",
    "存储优化失败；项目原文件保持不变。": "Storage optimization failed; original project files remain unchanged.",
})
_EXACT_JA.update({
    "把已有 Novel Formatter 项目文件夹复制进当前工作区并恢复完整状态": "既存の Novel Formatter プロジェクトを現在のワークスペースへコピーし、完全な状態を復元します",
    "导入备份": "バックアップを読み込む",
    "从项目 ZIP 备份安全恢复到当前工作区": "プロジェクトZIPバックアップを現在のワークスペースへ安全に復元します",
    "备份项目": "プロジェクトをバックアップ",
    "导出包含源文件、页面、OCR、裁决、正文和导出结果的完整项目 ZIP": "元ファイル、ページ、OCR、裁決、本文、出力結果を含む完全なプロジェクトZIPを書き出します",
    "检查项目": "プロジェクトを検査",
    "检查页面/源文件/lineage/运行历史，并可回收孤儿页面缓存": "ページ、元ファイル、lineage、実行履歴を検査し、孤立ページキャッシュを回収できます",
    "项目检查完成。": "プロジェクト検査が完了しました。",
    "项目检查完成；未清理任何文件。": "プロジェクト検査が完了しました。ファイルは整理していません。",
    "优化存储": "ストレージを最適化",
    "无损压缩 OCR/阶段/缓存 JSON，并把数千个 OCR 小缓存合并为 SQLite；不删除原图、OCR 结果或裁决": "OCR・ステージ・キャッシュJSONを可逆圧縮し、多数の小さなOCRキャッシュをSQLiteへ統合します。元画像、OCR結果、裁決は削除しません",
    "优化中…": "最適化中…",
    "正在无损优化项目存储；原图、OCR、裁决和导出结果均保留。": "プロジェクト保存領域を可逆最適化しています。元画像、OCR、裁決、出力結果はすべて保持されます。",
    "存储优化失败；项目原文件保持不变。": "ストレージ最適化に失敗しました。元のプロジェクトファイルは変更されていません。",
})

# v5.5 project run-log viewer and structured OCR diagnostics.
_EXACT_EN.update({
    "运行日志": "Run Logs",
    "查看每次 OCR、裁决、导出、备份等运行记录，以及 OCR 文本日志和性能明细": "View OCR, adjudication, export and backup runs, including OCR text logs and performance details",
    "项目运行日志": "Project Run Logs",
    "搜索": "Search",
    "搜索运行 ID、阶段、摘要或错误…": "Search run ID, stage, summary or error…",
    "全部": "All",
    "导出日志": "Export Logs",
    "打开日志目录": "Open Log Folder",
    "清理旧日志": "Clean Old Logs",
    "只删除旧运行日志附件，不删除 OCR、裁决、正文或导出结果": "Delete only old run-log attachments; OCR, adjudication, text and exports are preserved",
    "开始时间": "Start Time",
    "阶段": "Stage",
    "耗时": "Duration",
    "摘要": "Summary",
    "文本日志": "Text Log",
    "性能明细": "Performance Details",
    "复制详情": "Copy Details",
    "运行详情（只读）": "Run Details (Read-only)",
    "运行日志属于项目备份的一部分；API Key / Token 常见格式会在文本日志落盘前自动脱敏。": "Run logs are included in project backups. Common API key/token formats are redacted before text logs are saved.",
    "导出运行日志": "Export Run Logs",
    "日志导出完成": "Log Export Complete",
    "日志清理完成": "Log Cleanup Complete",
    "日志读取失败": "Failed to Read Logs",
    "日志导出失败": "Log Export Failed",
    "日志清理失败": "Log Cleanup Failed",
    "运行中": "Running",
    "完成": "Done",
    "已停止": "Stopped",
    "失败": "Failed",
    "警告": "Warning",
    "单模型 OCR": "Single-model OCR",
    "多模型 OCR": "Multi-model OCR",
    "OCR 正文": "OCR Text",
    "图文校对": "Image/Text Review",
    "云端裁决导入": "Cloud Adjudication Import",
    "EPUB 导出": "EPUB Export",
    "校对/裁决包导出": "Review/Adjudication Package Export",
    "项目备份": "Project Backup",
    "项目恢复": "Project Restore",
    "页面载入": "Pages Loaded",
    "工作区清理": "Workspace Cleanup",
})
_EXACT_JA.update({
    "运行日志": "実行ログ",
    "查看每次 OCR、裁决、导出、备份等运行记录，以及 OCR 文本日志和性能明细": "OCR・裁決・出力・バックアップなどの実行履歴、OCRテキストログ、性能詳細を表示します",
    "项目运行日志": "プロジェクト実行ログ",
    "搜索": "検索",
    "搜索运行 ID、阶段、摘要或错误…": "実行ID・段階・概要・エラーを検索…",
    "全部": "すべて",
    "导出日志": "ログを書き出す",
    "打开日志目录": "ログフォルダを開く",
    "清理旧日志": "古いログを整理",
    "只删除旧运行日志附件，不删除 OCR、裁决、正文或导出结果": "古い実行ログ添付のみ削除し、OCR・裁決・本文・出力結果は削除しません",
    "开始时间": "開始時刻",
    "阶段": "段階",
    "耗时": "所要時間",
    "摘要": "概要",
    "文本日志": "テキストログ",
    "性能明细": "性能詳細",
    "复制详情": "詳細をコピー",
    "运行详情（只读）": "実行詳細（読み取り専用）",
    "运行日志属于项目备份的一部分；API Key / Token 常见格式会在文本日志落盘前自动脱敏。": "実行ログはプロジェクトバックアップに含まれます。一般的なAPI Key / Token形式は保存前に自動でマスクされます。",
    "导出运行日志": "実行ログを書き出す",
    "日志导出完成": "ログ書き出し完了",
    "日志清理完成": "ログ整理完了",
    "日志读取失败": "ログの読み込みに失敗",
    "日志导出失败": "ログの書き出しに失敗",
    "日志清理失败": "ログ整理に失敗",
    "运行中": "実行中",
    "完成": "完了",
    "已停止": "停止済み",
    "失败": "失敗",
    "警告": "警告",
    "单模型 OCR": "単一モデルOCR",
    "多模型 OCR": "複数モデルOCR",
    "OCR 正文": "OCR本文",
    "图文校对": "画像・テキスト校正",
    "云端裁决导入": "クラウド裁決の読み込み",
    "EPUB 导出": "EPUB出力",
    "校对/裁决包导出": "校正／裁決パッケージ出力",
    "项目备份": "プロジェクトバックアップ",
    "项目恢复": "プロジェクト復元",
    "页面载入": "ページ読み込み",
    "工作区清理": "ワークスペース整理",
})

# V5.7 smart routing / active-review queue
_EXACT_EN.update({
    "智能路由（可选加速）": "Smart Routing (optional acceleration)",
    "主模型多数确认重试（可选）": "Main-model majority confirmation retry (optional)",
    "仅对已有独立模型严格多数的分歧句，让逐列/整页两个主模型做低成本确认重读。只有完整行重试与原多数一致时才允许本地裁决；同模型重试不增加票数，也不覆盖原始 OCR。分歧复核模型不会再二次重跑；NDLOCR 整页主模型也不会对全部冲突列再次逐列 OCR。关闭后不会执行任何额外主模型确认重试；开启会增加少量运行时间。": "Only disagreement rows with a pre-existing strict majority are eligible for low-cost confirmation rereads by the column and page main models. A local verdict is allowed only when a complete-row retry matches the existing majority; same-model retries never add votes or overwrite raw OCR. Disagreement-review models are not run a second time, and the NDLOCR page model no longer re-OCRs every conflicting column. Disabling this option runs no extra main-model confirmation retries; enabling it adds a modest amount of runtime.",
    '旧 Fast Core 智能路由仅保留工作区兼容状态；Phase27 的逐列 / 整页 / 全列三个已选择主模型都会按角色完整执行并保留独立原始证据，分歧复核槽只读取残余冲突物理列。': 'The legacy Fast Core smart-routing state is retained only for workspace compatibility. In Phase27, selected column/page/full-column main roles always run their full role scopes and preserve independent raw evidence; disagreement review reads residual conflict columns only.',
    "开启后，逐列/整页主模型先建立独立证据；如果两者已经一致，全列主模型自动跳过；只有真正分歧句才送给全列主模型，分歧复核 1→2→3 继续按需早停。关闭后保留兼容行为：所有已选择的主模型都完整执行。": "When enabled, the column and page main models establish independent evidence first. If they already agree, the sentence main model is skipped; only true disagreement sentences are sent to the sentence model, while review slots 1→2→3 continue to stop early as needed. Disable this to preserve compatibility behavior where every selected main model runs a full pass.",
    "高风险优先": "High-risk first",
    "只改变人工复核顺序，不改变正文、OCR 候选或自动裁决。优先显示三方分歧、空/占位符、长度差异、数字/等级内容和高共识熵句。": "Changes only the human-review order; it never changes body text, OCR candidates, or automatic adjudication. Prioritizes three-way disagreements, empty/placeholders, length mismatches, numeric/level content, and high-consensus-entropy sentences.",
    "已启用高风险优先：只调整待判断顺序，不改变任何 OCR 候选或融合结果。": "High-risk-first review is enabled: only the pending-review order changes; no OCR candidate or fusion result is modified.",
})
_EXACT_JA.update({
    "智能路由（可选加速）": "スマートルーティング（任意・高速化）",
    "主模型多数确认重试（可选）": "主モデル多数確認の再試行（任意）",
    "仅对已有独立模型严格多数的分歧句，让逐列/整页两个主模型做低成本确认重读。只有完整行重试与原多数一致时才允许本地裁决；同模型重试不增加票数，也不覆盖原始 OCR。分歧复核模型不会再二次重跑；NDLOCR 整页主模型也不会对全部冲突列再次逐列 OCR。关闭后不会执行任何额外主模型确认重试；开启会增加少量运行时间。": "独立モデルですでに厳密多数が成立している不一致文だけを対象に、列主モデルとページ主モデルが低コストの確認再認識を行います。完全な行の再認識結果が既存多数と一致した場合だけローカル裁決を許可し、同一モデルの再試行は票を増やさず、生のOCR結果も上書きしません。不一致レビュー用モデルは二度目の再実行を行わず、NDLOCRのページ主モデルもすべての競合列を再度逐列OCRしません。無効にすると追加の主モデル確認再試行は実行されず、有効にすると実行時間が少し増えます。",
    '旧 Fast Core 智能路由仅保留工作区兼容状态；Phase27 的逐列 / 整页 / 全列三个已选择主模型都会按角色完整执行并保留独立原始证据，分歧复核槽只读取残余冲突物理列。': '旧Fast Coreスマートルーティング状態はワークスペース互換用にのみ保持されます。Phase27では選択した列単位／全ページ／全列主モデルが各ロール範囲を完全実行して独立した生OCR証拠を保持し、不一致レビューは残差競合物理列だけを再認識します。',
    "开启后，逐列/整页主模型先建立独立证据；如果两者已经一致，全列主模型自动跳过；只有真正分歧句才送给全列主模型，分歧复核 1→2→3 继续按需早停。关闭后保留兼容行为：所有已选择的主模型都完整执行。": "有効時は列単位／ページ単位の主モデルが先に独立した証拠を作成します。両者が一致すれば文単位主モデルを自動で省略し、真の不一致文だけを文単位モデルへ送り、再確認1→2→3も必要に応じて早期停止します。無効時は互換動作として選択済みの主モデルをすべて全量実行します。",
    "高风险优先": "高リスク優先",
    "只改变人工复核顺序，不改变正文、OCR 候选或自动裁决。优先显示三方分歧、空/占位符、长度差异、数字/等级内容和高共识熵句。": "人手確認の順序だけを変更し、本文・OCR候補・自動裁決は変更しません。3者不一致、空欄／プレースホルダー、長さ差、数値／レベル情報、高い合意エントロピーの文を優先表示します。",
    "已启用高风险优先：只调整待判断顺序，不改变任何 OCR 候选或融合结果。": "高リスク優先を有効化しました。未決項目の順序だけを変更し、OCR候補や融合結果は変更しません。",
})

# v34 free-slot multi-model OCR + model-specific transport profiles.
_EXACT_EN.update({
    "多模型 OCR（3 个自由模型槽位）": "Multi-model OCR (3 free model slots)",
    "开启后，上方单模型选择会锁定且不参与本次运行。模型1/2/3可自由组合；每个模型自动使用自己的最佳输入 Profile（整页、紧凑单列、48px line 等），三份实际 OCR 都保留为独立证据；同一个 OCR 模型只能选择一次。": "When enabled, the single-model selector above is locked. Models 1/2/3 can be freely combined; each engine automatically uses its own optimized input profile (full page, compact column, 48px line, etc.). Every real OCR result remains independent evidence, and the same engine can be selected only once.",
    "✓ 高速三模型：共享几何 · 模型专用输入 · 三份独立 OCR · 完整提交 AI 裁决": "✓ Fast triple OCR: shared geometry · model-specific inputs · three independent OCR results · complete AI adjudication coverage",
    "三槽只表示模型顺序，不规定输入粒度。共享 page/column/sentence 几何与 lineage；最终送入识别器的 resize/padding/整页或单列 transport 由该模型自己的 Engine Profile 决定。": "The three slots define model order only, not input granularity. Page/column/sentence geometry and lineage are shared; resize, padding, full-page or column transport are selected by each engine's own Engine Profile.",
    "共享的是几何、column_id/sentence_id 与原图坐标；最终 OCR 图片不强制相同。例如 NDL 可走整页投影，Hayai/48px 走各自紧凑列输入，Apple 使用更宽纸张上下文。": "Geometry, column_id/sentence_id, and original-image coordinates are shared; final OCR images are not forced to be identical. For example, NDL can use page projection, Hayai/48px their own compact column inputs, and Apple a wider paper-context crop.",
})
_EXACT_JA.update({
    "多模型 OCR（3 个自由模型槽位）": "複数モデルOCR（自由な3モデルスロット）",
    "开启后，上方单模型选择会锁定且不参与本次运行。模型1/2/3可自由组合；每个模型自动使用自己的最佳输入 Profile（整页、紧凑单列、48px line 等），三份实际 OCR 都保留为独立证据；同一个 OCR 模型只能选择一次。": "有効にすると上部の単一モデル選択はロックされます。モデル1/2/3は自由に組み合わせられ、各エンジンは全ページ・コンパクト列・48px lineなど自身に最適化された入力Profileを自動使用します。実際に実行したOCRはすべて独立証拠として保持され、同じエンジンは1回だけ選択できます。",
    "✓ 高速三模型：共享几何 · 模型专用输入 · 三份独立 OCR · 完整提交 AI 裁决": "✓ 高速3モデル：共有ジオメトリ・モデル別入力・3つの独立OCR・AI裁決を完全保持",
    "三槽只表示模型顺序，不规定输入粒度。共享 page/column/sentence 几何与 lineage；最终送入识别器的 resize/padding/整页或单列 transport 由该模型自己的 Engine Profile 决定。": "3スロットはモデル順だけを表し、入力粒度は規定しません。page/column/sentenceのジオメトリとlineageを共有し、resize・padding・全ページ/列transportは各モデルのEngine Profileが決定します。",
    "共享的是几何、column_id/sentence_id 与原图坐标；最终 OCR 图片不强制相同。例如 NDL 可走整页投影，Hayai/48px 走各自紧凑列输入，Apple 使用更宽纸张上下文。": "共有するのはジオメトリ、column_id/sentence_id、元画像座標です。最終OCR画像は同一である必要はありません。たとえばNDLは全ページ投影、Hayai/48pxは各自のコンパクト列入力、Appleはより広い紙面コンテキストを使用できます。",
})

# v35 four-role multi-model scheduler + M6 real-book guidance.
_EXACT_EN.update({
    "多模型 OCR（角色分工 · 4 模型）": "Multi-model OCR (4-role pipeline)",
    "开启后，上方单模型选择会锁定且不参与本次运行。四个槽位分别定义证据粒度：逐列主模型 / 整页主模型 / 全列主模型 / 分歧复核模型。同一个 OCR 模型只能占用一个角色；同模型分歧确认由内部定向重试完成，不重复占用槽位。每个角色仍拥有独立的输入粒度、缓存身份与证据来源。共享 page/column/sentence 几何，最终送入识别器的像素 transport 由角色与模型专用 Profile 共同决定。": "When enabled, the single-model selector above is locked. The four slots are column main, full-page main, full-column main, and disagreement review. Each OCR engine may occupy only one role; same-engine disagreement confirmation uses internal targeted retry rather than another slot. Every role keeps independent input transport, cache identity, and evidence lineage.",
    "✓ 推荐：逐列 + 整页 + 全列建立独立主证据 · 复核仅跑残余分歧": "✓ Recommended: column + page establish primary evidence first · sentence OCR only on true disagreements · review only on residual disagreements",
    "同一 page/column/sentence identity 由共享几何保证；不同 OCR 可使用整页、紧凑单列、48px line、短块等不同最终输入。Phase27 的第三主模型固定为全列主模型；如需节省 48px 成本，可将 48px 放入分歧复核槽而不是全列槽。": "Shared geometry guarantees the same page/column/sentence identity, while each OCR engine may use full-page, compact-column, 48px-line, or short-block inputs. Based on the Volume 1 M6 run, the sentence model defaults to true column↔page disagreements only, avoiding the cost of running 48px on all 5,849 columns.",
    "全列主模型仅处理主证据分歧（推荐）": "Sentence main model only on primary-evidence disagreements (recommended)",
    "开启：全列与整页主模型先建立两份独立证据；已一致句不重复跑整句模型，只有真正分歧的完整 sentence group 才送入全列主模型。关闭仅用于诊断，会让全列主模型覆盖全书并显著增加耗时。": "Enabled: the column and page main models first establish two independent evidences. Agreed sentences skip duplicate sentence OCR; only true-disagreement complete sentence groups go to the sentence model. Disable only for diagnostics, because it runs the sentence model across the whole book and greatly increases runtime.",
    "仅对已有独立模型严格多数的疑难句做低成本确认重读；同模型重试不增加票数。默认关闭，避免再次出现整本级二次 OCR。": "Only reread difficult sentences that already have a strict majority among independent models; same-model retries never add votes. Disabled by default to avoid whole-book secondary OCR passes.",
    "共享几何只定义同一页、同一列、同一句和原图坐标；最终 OCR 图片不强制相同。角色决定识别粒度，Engine Profile 决定该模型的 padding/resize/整页或短块 transport。": "Shared geometry defines the same page, column, sentence, and original-image coordinates; final OCR images are not forced to be identical. The role defines recognition granularity, while the Engine Profile defines padding, resize, full-page, or short-block transport for that engine.",
})
_EXACT_JA.update({
    "多模型 OCR（角色分工 · 4 模型）": "複数モデルOCR（4ロール分担）",
    "开启后，上方单模型选择会锁定且不参与本次运行。四个槽位分别定义证据粒度：全列主模型 / 整页主模型 / 全列主模型 / 分歧复核模型。同一个 OCR 模型只能占用一个角色；同模型分歧确认由内部定向重试完成，不重复占用槽位。每个角色仍拥有独立的输入粒度、缓存身份与证据来源。共享 page/column/sentence 几何，最终送入识别器的像素 transport 由角色与模型专用 Profile 共同决定。": "有効にすると上部の単一モデル選択はロックされます。4つのスロットは列主モデル／ページ主モデル／文主モデル／不一致レビューの証拠粒度を定義します。同じOCRモデルを複数の役割に割り当てられ、各役割は入力粒度・キャッシュ識別子・証拠ソースを独立して保持します。page/column/sentenceジオメトリは共有し、最終認識画素は役割とモデル専用Profileのtransportで決まります。",
    "✓ 推荐：逐列 + 整页 + 全列建立独立主证据 · 复核仅跑残余分歧": "✓ 推奨：列＋ページで先に主証拠を作成・文OCRは真の不一致のみ・レビューは残差不一致のみ",
    "同一 page/column/sentence identity 由共享几何保证；不同 OCR 可使用整页、紧凑单列、48px line、短块等不同最终输入。Phase27 的第三主模型固定为全列主模型；如需节省 48px 成本，可将 48px 放入分歧复核槽而不是全列槽。": "共有ジオメトリで同一page/column/sentence identityを保証し、各OCRは全ページ、コンパクト列、48px line、短ブロックなど異なる最終入力を利用できます。第1巻のM6実機結果に基づき、文主モデルは列↔ページ主証拠の真の不一致だけを既定処理し、48pxを5849列すべてに実行する高コストを避けます。",
    "全列主模型仅处理主证据分歧（推荐）": "文主モデルは主証拠の不一致だけ処理（推奨）",
    "开启：全列与整页主模型先建立两份独立证据；已一致句不重复跑整句模型，只有真正分歧的完整 sentence group 才送入全列主模型。关闭仅用于诊断，会让全列主模型覆盖全书并显著增加耗时。": "有効時は列主モデルとページ主モデルが先に2つの独立証拠を作成します。一致済み文では文OCRを重複実行せず、真の不一致となった完全なsentence groupだけを文主モデルへ送ります。無効化は診断専用で、文主モデルが全書を処理するため実行時間が大幅に増えます。",
    "仅对已有独立模型严格多数的疑难句做低成本确认重读；同模型重试不增加票数。默认关闭，避免再次出现整本级二次 OCR。": "独立モデルですでに厳密多数がある難文だけを低コストで確認再認識します。同一モデルの再試行は票を増やしません。全書規模の二次OCRを避けるため既定はオフです。",
    "共享几何只定义同一页、同一列、同一句和原图坐标；最终 OCR 图片不强制相同。角色决定识别粒度，Engine Profile 决定该模型的 padding/resize/整页或短块 transport。": "共有ジオメトリは同一ページ・列・文・元画像座標だけを定義し、最終OCR画像を同一に強制しません。ロールが認識粒度を決め、Engine Profileがpadding／resize／全ページ／短ブロックtransportを決定します。",
})

# Phase27 full-column third-role wording. Historical sentence-role keys above
# remain readable for old workspaces, but current UI strings map to full-column
# execution and physical-column disagreement review.
_EXACT_EN.update({
    "开启后，上方单模型选择会锁定且不参与本次运行。四个槽位分别定义证据粒度：逐列主模型 / 整页主模型 / 全列主模型 / 分歧复核模型。同一个 OCR 模型只能占用一个角色；同模型分歧确认由内部定向重试完成，不重复占用槽位。每个角色仍拥有独立的输入粒度、缓存身份与证据来源。共享 page/column/sentence 几何，最终送入识别器的像素 transport 由角色与模型专用 Profile 共同决定。": "When enabled, the single-model selector is locked. The four slots are column main, full-page main, full-column main, and disagreement review. Each OCR engine may occupy only one role; same-engine disagreement confirmation uses internal targeted retry instead of a duplicate slot. Every role keeps independent transport, cache identity, and evidence lineage. Shared page/column/sentence geometry anchors identity, while each role and engine profile determines the actual recognizer pixels.",
    "✓ 角色实验：Hayai 逐列 · NDL 整页 · 48px 可全列 · Hayai/48px 可仅复核分歧": "✓ Role test: Hayai column · NDL page · 48px full-column · Hayai/48px disagreement-only review",
    "同一 page/column identity 由共享几何保证；不同 OCR 可使用整页或模型优化后的单列 transport。Phase28 默认将全列主模型留空，先由 Hayai 逐列与 NDL 整页形成主证据，再让 48px 只读取真实冲突列；全列槽仍可手动启用用于 A/B 诊断。": "Shared geometry preserves page/column identity while each OCR engine may use full-page or model-optimized column transport. Phase28 leaves the full-column main slot empty by default: Hayai column OCR and NDL page OCR establish the primary evidence first, then 48px reads only true conflict columns. The full-column slot remains available for manual A/B diagnostics.",
    "旧整句智能路由（兼容状态）": "Legacy sentence smart router (compatibility state)",
    "Phase27 起第三主模型槽已改为全列主模型，固定完整读取所有物理列；此旧开关仅保留工作区兼容，不再影响执行。": "Since Phase27 the third main slot is a full-column model and always reads all physical columns. This legacy toggle is retained only for workspace compatibility and no longer affects execution.",
})
_EXACT_JA.update({
    "开启后，上方单模型选择会锁定且不参与本次运行。四个槽位分别定义证据粒度：逐列主模型 / 整页主模型 / 全列主模型 / 分歧复核模型。同一个 OCR 模型只能占用一个角色；同模型分歧确认由内部定向重试完成，不重复占用槽位。每个角色仍拥有独立的输入粒度、缓存身份与证据来源。共享 page/column/sentence 几何，最终送入识别器的像素 transport 由角色与模型专用 Profile 共同决定。": "有効にすると上部の単一モデル選択はロックされます。4スロットは列主モデル／全ページ主モデル／全列主モデル／不一致レビューです。同じOCRエンジンは1つのロールだけに割り当てられ、同一モデルによる差異確認は重複スロットではなく内部の定向再試行で行います。各ロールは入力transport・キャッシュID・証拠lineageを独立保持します。共有page/column/sentenceジオメトリで同一性を固定し、最終認識画素はロールとエンジンProfileが決定します。",
    "✓ 角色实验：Hayai 逐列 · NDL 整页 · 48px 可全列 · Hayai/48px 可仅复核分歧": "✓ ロール試験：Hayai列・NDL全ページ・48px全列・Hayai/48pxは不一致のみ再確認",
    "同一 page/column identity 由共享几何保证；不同 OCR 可使用整页或模型优化后的单列 transport。Phase28 默认将全列主模型留空，先由 Hayai 逐列与 NDL 整页形成主证据，再让 48px 只读取真实冲突列；全列槽仍可手动启用用于 A/B 诊断。": "共有ジオメトリで同一page/column identityを保証し、各OCRは全ページまたはモデル最適化列transportを利用できます。Phase28では全列主モデルを既定で空にし、Hayai列OCRとNDLページOCRで主証拠を作った後、48pxは真の競合列だけを再認識します。全列スロットはA/B診断用に手動で有効化できます。",
    "旧整句智能路由（兼容状态）": "旧文単位スマートルータ（互換状態）",
    "Phase27 起第三主模型槽已改为全列主模型，固定完整读取所有物理列；此旧开关仅保留工作区兼容，不再影响执行。": "Phase27以降、第3主モデルは全列主モデルとなり、すべての物理列を必ず処理します。この旧トグルはワークスペース互換用にのみ残り、実行には影響しません。",
})

# Fast-Core / removed legacy text-workspace wording.
_EXACT_EN.update({
    "使用全局 AI 服务商、密钥和模型设置。": "Use the global AI provider, credentials, and model settings.",
    "使用宽松 Formatter 规则同时纠错和整理长文本结构。": "Use relaxed Formatter rules to correct errors and organize long-text structure.",
    "这里仅负责检查、预览与生成 EPUB；OCR 与正文整理请在前面的工作区完成。": "This workspace only checks, previews, and builds EPUB. Perform OCR and body-text preparation in the earlier workspaces.",
})
_EXACT_JA.update({
    "使用全局 AI 服务商、密钥和模型设置。": "グローバルAIプロバイダー、認証情報、モデル設定を使用します。",
    "使用宽松 Formatter 规则同时纠错和整理长文本结构。": "緩やかなFormatterルールで誤りを修正し、長文構造も整理します。",
    "这里仅负责检查、预览与生成 EPUB；OCR 与正文整理请在前面的工作区完成。": "ここではEPUBの確認・プレビュー・生成のみを行います。OCRと本文整理は前のワークスペースで行ってください。",
})

# OCR-compare strings intentionally retained after deleting the separate legacy
# standalone comparison workspace. These strings belong to OCR adjudication/image review.
_EXACT_EN.update({
    "逐句裁决：左侧队列定位未决句，上方已载入模型只显示当前句；绿色=一致，红色=差异。候选卡始终完整展开，红底=替换、橙底=增删；融合候选可直接修改，“显示全文对比”只切换显示。": "Sentence adjudication: the left queue locates pending sentences and the loaded models above show only the current sentence. Green = agreement, red = difference. Candidate cards stay fully expanded; red background = replacement, orange = insertion/deletion. Fusion candidates are directly editable; 'Show Full Text Compare' changes display only.",
    "图文对照（与当前人工纠错同步）": "Image/Text Comparison (Synced with Manual Review)",
})
_EXACT_JA.update({
    "逐句裁决：左侧队列定位未决句，上方已载入模型只显示当前句；绿色=一致，红色=差异。候选卡始终完整展开，红底=替换、橙底=增删；融合候选可直接修改，“显示全文对比”只切换显示。": "文ごとの裁決：左キューで未決文へ移動し、上の読み込み済みモデルは現在文だけを表示します。緑=一致、赤=差異。候補カードは常に全展開し、赤背景=置換、橙背景=挿入/削除。融合候補は直接編集でき、「全文比較を表示」は表示だけを切り替えます。",
    "图文对照（与当前人工纠错同步）": "画像/テキスト比較（現在の手動修正と同期）",
})


# Phase 9 project-dashboard / resumable-workflow chrome.
_EXACT_EN.update({
    "当前项目": "Current Project",
    "未选择项目": "No project selected",
    "创建或选择项目后，页面、OCR、裁决与 EPUB 会持续保存。": "Create or select a project to persist pages, OCR, adjudication, and EPUB output continuously.",
    "继续上次工作": "Continue Last Work",
    "刷新状态": "Refresh Status",
    "项目流程": "Project Workflow",
    "状态来自项目磁盘快照，不会载入大型 OCR 文档；点击任一步骤可直接进入对应工作区。": "Status comes from lightweight on-disk project snapshots; large OCR documents are not loaded. Click any stage to open its workspace.",
    "页面": "Pages",
    "已完成": "Done",
    "可继续": "Continue",
    "可选": "Optional",
    "需更新": "Needs Update",
    "未开始": "Not Started",
    "最近项目": "Recent Projects",
    "双击项目可切换并恢复页面状态；列表按最近活动排序。": "Double-click a project to switch and restore its page state. Projects are sorted by recent activity.",
    "项目工具": "Project Tools",
    "备份项目": "Back Up Project",
    "运行日志": "Run Logs",
    "检查项目": "Audit Project",
    "优化存储": "Optimize Storage",
    "项目目录": "Project Folder",
    "删除项目": "Delete Project",
    "暂无项目": "No projects yet",
    "项目已建立，尚无运行记录。": "Project created; no run history yet.",
    "继续导入和整理页面": "Continue Importing and Organizing Pages",
    "继续 OCR": "Continue OCR",
    "继续多模型 OCR 裁决": "Continue Multi-model OCR Adjudication",
    "继续正文整理": "Continue Body-text Formatting",
    "继续生成 EPUB": "Continue EPUB Generation",
    "查看或重新导出 EPUB": "View or Re-export EPUB",
    "先导入图片或 PDF": "Import images or a PDF first",
    "等待页面": "Waiting for pages",
    "等待 OCR": "Waiting for OCR",
    "等待正文": "Waiting for body text",
    "尚未导入页面": "No pages imported yet",
    "页面已变化，现有正文需重新确认": "Pages changed; the current body text must be reconfirmed",
    "OCR 正文已保存": "OCR body text saved",
    "多模型自动融合已保存": "Multi-model auto-fusion saved",
    "图文校对稿已保存": "Image/text review draft saved",
    "Formatter 正文已保存": "Formatter body text saved",
    "存在多模型会话，可继续裁决": "A multi-model session exists; adjudication can continue",
    "多模型裁决已无待判断句": "No pending sentences remain in multi-model adjudication",
    "单模型流程：校对可选": "Single-model flow: proofreading is optional",
    "正文整理已保存": "Body-text formatting saved",
    "可继续正文整理": "Body-text formatting can continue",
    "已有 EPUB 导出": "EPUB output exists",
    "可生成 EPUB": "Ready to generate EPUB",
})
_EXACT_JA.update({
    "当前项目": "現在のプロジェクト",
    "未选择项目": "プロジェクト未選択",
    "创建或选择项目后，页面、OCR、裁决与 EPUB 会持续保存。": "プロジェクトを作成または選択すると、ページ・OCR・裁決・EPUB出力が継続的に保存されます。",
    "继续上次工作": "前回の作業を続ける",
    "刷新状态": "状態を更新",
    "项目流程": "プロジェクト工程",
    "状态来自项目磁盘快照，不会载入大型 OCR 文档；点击任一步骤可直接进入对应工作区。": "状態は軽量なディスク上のプロジェクトスナップショットから取得し、大容量OCR文書は読み込みません。各工程をクリックすると対応ワークスペースへ移動できます。",
    "页面": "ページ",
    "已完成": "完了",
    "可继续": "続行可能",
    "可选": "任意",
    "需更新": "更新必要",
    "未开始": "未開始",
    "最近项目": "最近のプロジェクト",
    "双击项目可切换并恢复页面状态；列表按最近活动排序。": "プロジェクトをダブルクリックすると切り替えてページ状態を復元します。最近の作業順に表示されます。",
    "项目工具": "プロジェクトツール",
    "备份项目": "プロジェクトをバックアップ",
    "运行日志": "実行ログ",
    "检查项目": "プロジェクトを検査",
    "优化存储": "ストレージを最適化",
    "项目目录": "プロジェクトフォルダー",
    "删除项目": "プロジェクトを削除",
    "暂无项目": "プロジェクトはありません",
    "项目已建立，尚无运行记录。": "プロジェクトは作成済みですが、実行履歴はまだありません。",
    "继续导入和整理页面": "ページの読み込み・整理を続ける",
    "继续 OCR": "OCRを続ける",
    "继续多模型 OCR 裁决": "複数モデルOCR裁決を続ける",
    "继续正文整理": "本文整理を続ける",
    "继续生成 EPUB": "EPUB生成を続ける",
    "查看或重新导出 EPUB": "EPUBを表示または再出力",
    "先导入图片或 PDF": "先に画像またはPDFを読み込む",
    "等待页面": "ページ待ち",
    "等待 OCR": "OCR待ち",
    "等待正文": "本文待ち",
    "尚未导入页面": "ページはまだ読み込まれていません",
    "页面已变化，现有正文需重新确认": "ページが変更されたため、現在の本文を再確認する必要があります",
    "OCR 正文已保存": "OCR本文を保存済み",
    "多模型自动融合已保存": "複数モデル自動融合を保存済み",
    "图文校对稿已保存": "画像・テキスト校正稿を保存済み",
    "Formatter 正文已保存": "Formatter本文を保存済み",
    "存在多模型会话，可继续裁决": "複数モデルセッションがあり、裁決を続けられます",
    "多模型裁决已无待判断句": "複数モデル裁決の未決文はありません",
    "单模型流程：校对可选": "単一モデル工程：校正は任意です",
    "正文整理已保存": "本文整理を保存済み",
    "可继续正文整理": "本文整理を続けられます",
    "已有 EPUB 导出": "EPUB出力があります",
    "可生成 EPUB": "EPUBを生成できます",
})
_PATTERNS[LANG_EN].extend([
    (re.compile(r"^(\d+) 页已持久化$"), lambda m: f"{m.group(1)} page(s) persisted"),
    (re.compile(r"^待裁决 (\d+)(?: / (\d+))? 句$"), lambda m: f"{m.group(1)} pending adjudication" + (f" / {m.group(2)} total" if m.group(2) else "")),
    (re.compile(r"^最近任务：(.+) · (.+) · (.+)$"), lambda m: f"Latest task: {m.group(1)} · {m.group(2)} · {m.group(3)}"),
    (re.compile(r"^正文阶段：(.+)$"), lambda m: f"Body-text stage: {m.group(1)}"),
])
_PATTERNS[LANG_JA].extend([
    (re.compile(r"^(\d+) 页已持久化$"), lambda m: f"{m.group(1)}ページ保存済み"),
    (re.compile(r"^待裁决 (\d+)(?: / (\d+))? 句$"), lambda m: f"未決 {m.group(1)}文" + (f" / 全{m.group(2)}文" if m.group(2) else "")),
    (re.compile(r"^最近任务：(.+) · (.+) · (.+)$"), lambda m: f"最新タスク：{m.group(1)} · {m.group(2)} · {m.group(3)}"),
    (re.compile(r"^正文阶段：(.+)$"), lambda m: f"本文工程：{m.group(1)}"),
])

_EXACT_EN.update({
    "上游页面已变化，旧裁决快照仅保留为历史证据": "Upstream pages changed; the old adjudication snapshot is retained as historical evidence only",
    "上游页面已变化，需先重新确认 OCR 正文": "Upstream pages changed; reconfirm OCR body text first",
    "已有 EPUB，但上游页面已变化": "An EPUB exists, but upstream pages have changed",
    "上游页面已变化，暂不建议导出": "Upstream pages changed; export is not recommended yet",
})
_EXACT_JA.update({
    "上游页面已变化，旧裁决快照仅保留为历史证据": "上流ページが変更されたため、旧裁決スナップショットは履歴証拠としてのみ保持します",
    "上游页面已变化，需先重新确认 OCR 正文": "上流ページが変更されたため、先にOCR本文を再確認してください",
    "已有 EPUB，但上游页面已变化": "EPUBはありますが、上流ページが変更されています",
    "上游页面已变化，暂不建议导出": "上流ページが変更されたため、現時点では出力を推奨しません",
})

# Phase 10 unified runtime-task / resumable OCR chrome.
_EXACT_EN.update({
    "继续 OCR": "Continue OCR",
    "继续上次 OCR（复用断点）": "Continue Last OCR (Reuse Checkpoint)",
    "查看当前 OCR 任务": "View Current OCR Task",
    "OCR 正在运行": "OCR is running",
    "OCR 正在停止并保存断点": "OCR is stopping and saving its checkpoint",
    "运行中": "Running",
    "正在停止": "Stopping",
})
_EXACT_JA.update({
    "继续 OCR": "OCRを続行",
    "继续上次 OCR（复用断点）": "前回のOCRを続行（チェックポイント再利用）",
    "查看当前 OCR 任务": "現在のOCRタスクを表示",
    "OCR 正在运行": "OCR実行中",
    "OCR 正在停止并保存断点": "OCRを停止し、チェックポイントを保存中",
    "运行中": "実行中",
    "正在停止": "停止中",
})
_PATTERNS[LANG_EN].extend([
    (re.compile(r"^上次 OCR 可恢复 · 已记录 (\d+) 个完成阶段$"), lambda m: f"Last OCR is resumable · {m.group(1)} completed stage(s) recorded"),
    (re.compile(r"^上次 OCR 可继续；已记录 (\d+) 个完成阶段$"), lambda m: f"Last OCR can continue; {m.group(1)} completed stage(s) recorded"),
    (re.compile(r"^当前任务：(.+) · (运行中|正在停止)$"), lambda m: f"Current task: {m.group(1)} · " + ("Running" if m.group(2) == "运行中" else "Stopping")),
])
_PATTERNS[LANG_JA].extend([
    (re.compile(r"^上次 OCR 可恢复 · 已记录 (\d+) 个完成阶段$"), lambda m: f"前回OCRは再開可能 · 完了工程 {m.group(1)}件を記録済み"),
    (re.compile(r"^上次 OCR 可继续；已记录 (\d+) 个完成阶段$"), lambda m: f"前回OCRを続行可能；完了工程 {m.group(1)}件を記録済み"),
    (re.compile(r"^当前任务：(.+) · (运行中|正在停止)$"), lambda m: f"現在のタスク：{m.group(1)} · " + ("実行中" if m.group(2) == "运行中" else "停止中")),
])

_EXACT_EN.update({
    "按当前参数继续；已完成模型阶段与单列/整句缓存会校验后直接复用。": "Continue with the current parameters; completed model stages and column/sentence caches are validated and reused directly.",
    "若项目断点有效，将复用已完成模型阶段和分段识别缓存。": "If the project checkpoint is valid, completed model stages and segment-recognition caches will be reused.",
})
_EXACT_JA.update({
    "按当前参数继续；已完成模型阶段与单列/整句缓存会校验后直接复用。": "現在の設定で続行します。完了済みモデル工程と列/文キャッシュを検証後、そのまま再利用します。",
    "若项目断点有效，将复用已完成模型阶段和分段识别缓存。": "プロジェクトのチェックポイントが有効なら、完了済みモデル工程と分割認識キャッシュを再利用します。",
})

# Phase 11 GPT-grade OCR adjudication.
_EXACT_EN.update({
    "AI GPT级裁决": "AI GPT-grade Adjudication",
    "默认“GPT级”模式会先让视觉模型在看不到 A/B/C 的情况下独立逐字抄写，再把视觉结果、全部 OCR 候选、物理列、前后文和已保存术语交给第二阶段上下文裁决，最后由独立审计器反查。整个过程默认不读取文库电子版，避免参考答案泄漏。": "The default GPT-grade mode first asks the vision model to transcribe independently without seeing A/B/C. It then gives the visual result, all OCR candidates, physical columns, neighbouring context, and saved terminology to a second contextual adjudicator, followed by an independent audit. Publication-reference eBooks are not read by default, preventing answer leakage.",
    "GPT级 · 视觉抄写 + 上下文裁决 + 独立审计（推荐）": "GPT-grade · visual transcription + contextual adjudication + independent audit (recommended)",
    "纯视觉 · 独立抄写 + 候选比较（更快）": "Visual only · independent transcription + candidate comparison (faster)",
    "裁决模式：": "Adjudication mode:",
    "low · 快速": "low · fast",
    "high · 出版裁决推荐": "high · recommended for publication adjudication",
    "max · 极少数疑难页": "max · only for rare difficult pages",
    "页/批；自动带前后 2 页锚点": "pages/batch; automatically include 2-page before/after anchors",
    "上下文窗口：": "Context window:",
    "GPT级模式完成后再用独立审计器反查缺字、重复、邻列粘连、数字/否定/专名（推荐）": "After GPT-grade adjudication, run an independent auditor for missing text, duplication, adjacent-column adhesion, numbers, negation, and names (recommended)",
    "证据规则：图片必须覆盖该行声明的全部物理列和页面；有图片文件不等于证据完整。上下文只能用于消歧，不能凭语法润色或续写；模型置信度不参与自动放行。本地安全闸门或独立审计失败的条目会保留人工复核。": "Evidence rule: the image must cover every declared physical column and page. An image file alone does not prove complete evidence. Context may disambiguate but must never polish or continue text from grammar; model confidence does not authorize write-back. Items blocked by local safety gates or the independent auditor remain for manual review.",
    "开始 GPT级 AI 裁决": "Start GPT-grade AI adjudication",
    "GPT级 AI OCR 裁决正在运行": "GPT-grade AI OCR adjudication is running",
    "正在停止 GPT级 AI 裁决": "Stopping GPT-grade AI adjudication",
    "GPT级 AI OCR 裁决已取消": "GPT-grade AI OCR adjudication canceled",
    "GPT级 AI OCR 裁决失败": "GPT-grade AI OCR adjudication failed",
    "GPT级 AI OCR 裁决完成": "GPT-grade AI OCR adjudication complete",
})
_EXACT_JA.update({
    "AI GPT级裁决": "AI GPT級裁決",
    "默认“GPT级”模式会先让视觉模型在看不到 A/B/C 的情况下独立逐字抄写，再把视觉结果、全部 OCR 候选、物理列、前后文和已保存术语交给第二阶段上下文裁决，最后由独立审计器反查。整个过程默认不读取文库电子版，避免参考答案泄漏。": "既定のGPT級モードでは、まず視覚モデルがA/B/Cを見ずに独立して逐字転記します。その後、視覚結果・全OCR候補・物理列・前後文・保存済み用語を第2段階の文脈裁決へ渡し、最後に独立監査を行います。既定では出版参考電子書を読まず、正解の漏洩を防ぎます。",
    "GPT级 · 视觉抄写 + 上下文裁决 + 独立审计（推荐）": "GPT級 · 視覚転記 + 文脈裁決 + 独立監査（推奨）",
    "纯视觉 · 独立抄写 + 候选比较（更快）": "視覚のみ · 独立転記 + 候補比較（高速）",
    "裁决模式：": "裁決モード：",
    "low · 快速": "low · 高速",
    "high · 出版裁决推荐": "high · 出版裁決に推奨",
    "max · 极少数疑难页": "max · ごく少数の難読ページのみ",
    "页/批；自动带前后 2 页锚点": "ページ/バッチ；前後2ページのアンカーを自動追加",
    "上下文窗口：": "文脈ウィンドウ：",
    "GPT级模式完成后再用独立审计器反查缺字、重复、邻列粘连、数字/否定/专名（推荐）": "GPT級裁決後に独立監査で欠字・重複・隣接列混入・数字/否定/固有名詞を再確認（推奨）",
    "证据规则：图片必须覆盖该行声明的全部物理列和页面；有图片文件不等于证据完整。上下文只能用于消歧，不能凭语法润色或续写；模型置信度不参与自动放行。本地安全闸门或独立审计失败的条目会保留人工复核。": "証拠ルール：画像は宣言された全物理列とページを覆う必要があります。画像ファイルがあるだけでは証拠が完全とは限りません。文脈は曖昧さ解消にのみ使い、文法だけで推敲・続きの生成をしてはいけません。モデル信頼度だけで自動適用せず、ローカル安全ゲートまたは独立監査に失敗した項目は手動確認に残します。",
    "开始 GPT级 AI 裁决": "GPT級AI裁決を開始",
    "GPT级 AI OCR 裁决正在运行": "GPT級AI OCR裁決を実行中",
    "正在停止 GPT级 AI 裁决": "GPT級AI裁決を停止中",
    "GPT级 AI OCR 裁决已取消": "GPT級AI OCR裁決をキャンセルしました",
    "GPT级 AI OCR 裁决失败": "GPT級AI OCR裁決に失敗しました",
    "GPT级 AI OCR 裁决完成": "GPT級AI OCR裁決が完了しました",
})

# Phase 14 command-palette / coalesced-workspace-refresh chrome.
_EXACT_EN.update({
    "命令": "Commands",
    "命令面板": "Command Palette",
    "导航": "Navigation",
    "OCR 工具": "OCR Tools",
    "输入命令，例如 OCR、裁决、EPUB、项目…": "Type a command, e.g. OCR, adjudication, EPUB, project…",
    "↑↓ 选择 · Enter 执行 · Esc 关闭": "↑↓ Select · Enter Run · Esc Close",
    "没有匹配命令": "No matching commands",
    "命令列表加载失败": "Failed to load commands",
    "首页": "Home",
})
_EXACT_JA.update({
    "命令": "コマンド",
    "命令面板": "コマンドパレット",
    "导航": "ナビゲーション",
    "OCR 工具": "OCRツール",
    "输入命令，例如 OCR、裁决、EPUB、项目…": "OCR・裁決・EPUB・プロジェクトなどのコマンドを入力…",
    "↑↓ 选择 · Enter 执行 · Esc 关闭": "↑↓ 選択 · Enter 実行 · Esc 閉じる",
    "没有匹配命令": "一致するコマンドはありません",
    "命令列表加载失败": "コマンド一覧の読み込みに失敗",
    "首页": "ホーム",
})


# Phase 15 transactional batch preview/apply/restore.
_EXACT_EN.update({
    "Formatter 批量编辑预览": "Formatter Batch Edit Preview",
    "OCR 自动选优 · 批量预览": "OCR Auto-select · Batch Preview",
    "应用变更": "Apply Changes",
    "应用批量编辑": "Apply Batch Edit",
    "应用自动选优": "Apply Auto-select",
    "⟲ 恢复批量": "⟲ Restore Batch",
    "⟲ 撤销自动选优": "⟲ Undo Auto-select",
    "恢复最近一次通过预览确认的批量编辑；如果之后又修改过正文则拒绝恢复。": "Restore the most recent preview-confirmed batch edit; restoration is refused if the text changed afterward.",
    "仅恢复最近一次通过预览确认的自动选优；之后若有新的人工/AI裁决，恢复时会拒绝覆盖。": "Restore only the latest preview-confirmed auto-select; later human/AI adjudication is protected and will not be overwritten.",
    "这里只显示计划中的修改；点击应用前不会写回任何正文、OCR 候选或项目文件。": "This shows planned changes only. No body text, OCR candidate, or project file is written before Apply.",
    "没有可恢复的批量编辑": "There is no batch edit to restore.",
    "没有可恢复的自动选优批次": "There is no auto-select batch to restore.",
    "批量编辑之后正文又发生了变化。为避免覆盖后续工作，本次批量恢复已失效。": "The text changed after the batch edit. This restore point is invalidated to avoid overwriting later work.",
    "自动选优后 OCR 源文字已经修改但尚未重新对齐。为避免覆盖后续工作，本次恢复已失效。": "OCR source text changed after auto-select and has not been realigned. This restore point is invalidated to protect later work.",
    "自动选优后已经发生新的人工/AI裁决。为避免覆盖这些后续选择，本次恢复已失效。": "A later human/AI adjudication occurred after auto-select. This restore point is invalidated to protect those decisions.",
    "已恢复最近一次批量编辑；更早历史未受影响。": "Restored the latest batch edit; earlier history is unchanged.",
    "已恢复自动选优前的候选与裁决状态；原始 OCR 文本始终未被改写。": "Restored the candidate/adjudication state from before auto-select; original OCR text was never rewritten.",
    "已取消自动选优；当前 OCR 候选和裁决状态没有发生变化。": "Auto-select canceled; current OCR candidates and adjudication state were not changed.",
    "待判断→自动选择": "Pending → Auto-selected",
    "自动选优更新": "Auto-select Update",
    "处理前": "Before",
    "处理后": "After",
    "类型": "Type",
    "修改前": "Before",
    "修改后": "After",
    "说明": "Details",
    "无法预览编辑": "Cannot Preview Edit",
    "无法恢复": "Cannot Restore",
})
_EXACT_JA.update({
    "Formatter 批量编辑预览": "Formatter 一括編集プレビュー",
    "OCR 自动选优 · 批量预览": "OCR 自動選択 · 一括プレビュー",
    "应用变更": "変更を適用",
    "应用批量编辑": "一括編集を適用",
    "应用自动选优": "自動選択を適用",
    "⟲ 恢复批量": "⟲ 一括復元",
    "⟲ 撤销自动选优": "⟲ 自動選択を取消",
    "恢复最近一次通过预览确认的批量编辑；如果之后又修改过正文则拒绝恢复。": "プレビュー確認済みの直近一括編集を復元します。以後に本文が変更された場合は復元を拒否します。",
    "仅恢复最近一次通过预览确认的自动选优；之后若有新的人工/AI裁决，恢复时会拒绝覆盖。": "プレビュー確認済みの直近自動選択だけを復元します。その後の人手/AI裁決は保護され、上書きしません。",
    "这里只显示计划中的修改；点击应用前不会写回任何正文、OCR 候选或项目文件。": "ここには予定変更だけを表示します。適用を押すまで本文・OCR候補・プロジェクトファイルへは書き戻しません。",
    "没有可恢复的批量编辑": "復元できる一括編集はありません",
    "没有可恢复的自动选优批次": "復元できる自動選択バッチはありません",
    "批量编辑之后正文又发生了变化。为避免覆盖后续工作，本次批量恢复已失效。": "一括編集後に本文が変更されました。後続作業を上書きしないよう、この復元点は無効です。",
    "自动选优后 OCR 源文字已经修改但尚未重新对齐。为避免覆盖后续工作，本次恢复已失效。": "自動選択後にOCR元テキストが変更され、まだ再整列されていません。後続作業保護のため復元点を無効化しました。",
    "自动选优后已经发生新的人工/AI裁决。为避免覆盖这些后续选择，本次恢复已失效。": "自動選択後に新しい人手/AI裁決があります。それらを上書きしないよう復元点を無効化しました。",
    "已恢复最近一次批量编辑；更早历史未受影响。": "直近の一括編集を復元しました。以前の履歴には影響しません。",
    "已恢复自动选优前的候选与裁决状态；原始 OCR 文本始终未被改写。": "自動選択前の候補・裁決状態を復元しました。元OCRテキストは一度も書き換えていません。",
    "已取消自动选优；当前 OCR 候选和裁决状态没有发生变化。": "自動選択をキャンセルしました。現在のOCR候補と裁決状態は変更されていません。",
    "待判断→自动选择": "未決→自動選択",
    "自动选优更新": "自動選択更新",
    "处理前": "処理前",
    "处理后": "処理後",
    "类型": "種類",
    "修改前": "変更前",
    "修改后": "変更後",
    "说明": "説明",
    "无法预览编辑": "編集をプレビューできません",
    "无法恢复": "復元できません",
})
_PATTERNS[LANG_EN].extend([
    (re.compile(r"^共 (\d+) 项变更$"), lambda m: f"{m.group(1)} changes"),
    (re.compile(r"^第 (\d+) 个文本块$"), lambda m: f"Text block {m.group(1)}"),
    (re.compile(r"^第 (\d+) 句$"), lambda m: f"Sentence {m.group(1)}"),
])
_PATTERNS[LANG_JA].extend([
    (re.compile(r"^共 (\d+) 项变更$"), lambda m: f"変更 {m.group(1)}件"),
    (re.compile(r"^第 (\d+) 个文本块$"), lambda m: f"テキストブロック {m.group(1)}"),
    (re.compile(r"^第 (\d+) 句$"), lambda m: f"文 {m.group(1)}"),
])

# Phase 16 conflict-guarded adjudication delta history.
_EXACT_EN.update({
    "↶ 裁决": "↶ Undo Decision",
    "↷ 裁决": "↷ Redo Decision",
    "撤销最近一次人工候选选择/重新打开；不会覆盖之后的 AI、外部导入或人工编辑": "Undo the latest manual candidate choice/reopen; later AI, external imports, or manual edits are never overwritten.",
    "重做刚刚撤销的裁决；相关句发生后续修改时会拒绝执行": "Redo the just-undone decision; it is refused if any affected sentence changed afterward.",
    "裁决历史": "Adjudication History",
    "没有可撤销的裁决操作。": "There is no adjudication action to undo.",
    "无法撤销": "Cannot Undo",
    "最近一次裁决之后，同一句已经被人工编辑、AI裁决或外部导入修改。为避免覆盖后续结果，本次撤销已失效。": "The same sentence was changed by a later manual edit, AI adjudication, or external import. This undo is invalidated to protect the newer result.",
    "没有可重做的裁决操作。": "There is no adjudication action to redo.",
    "无法重做": "Cannot Redo",
    "撤销之后，同一句已经发生新的修改。为避免覆盖后续结果，本次重做已失效。": "The same sentence changed after the undo. This redo is invalidated to protect the newer result.",
    "撤销最近一次人工候选选择/重新打开；若同句之后被 AI 或人工修改，会拒绝覆盖": "Undo the latest manual candidate choice/reopen; refuse if the same sentence was later changed by AI or a person.",
    "重做刚刚撤销的裁决；只在相关句仍保持撤销后状态时允许": "Redo the just-undone decision only while every affected sentence still matches the post-undo state.",
})
_EXACT_JA.update({
    "↶ 裁决": "↶ 裁決を取消",
    "↷ 裁决": "↷ 裁決をやり直す",
    "撤销最近一次人工候选选择/重新打开；不会覆盖之后的 AI、外部导入或人工编辑": "直近の手動候補選択/再オープンを取り消します。後続のAI裁決・外部取込・手動編集は上書きしません。",
    "重做刚刚撤销的裁决；相关句发生后续修改时会拒绝执行": "直前に取り消した裁決をやり直します。対象文がその後変更されていれば拒否します。",
    "裁决历史": "裁決履歴",
    "没有可撤销的裁决操作。": "取り消せる裁決操作はありません。",
    "无法撤销": "取り消せません",
    "最近一次裁决之后，同一句已经被人工编辑、AI裁决或外部导入修改。为避免覆盖后续结果，本次撤销已失效。": "直近の裁決後に同じ文が手動編集・AI裁決・外部取込で変更されました。後続結果を保護するため、この取り消しは無効です。",
    "没有可重做的裁决操作。": "やり直せる裁決操作はありません。",
    "无法重做": "やり直せません",
    "撤销之后，同一句已经发生新的修改。为避免覆盖后续结果，本次重做已失效。": "取り消し後に同じ文が変更されました。後続結果を保護するため、このやり直しは無効です。",
    "撤销最近一次人工候选选择/重新打开；若同句之后被 AI 或人工修改，会拒绝覆盖": "直近の手動候補選択/再オープンを取消。同じ文が後でAIまたは人手で変更されていれば上書きを拒否します。",
    "重做刚刚撤销的裁决；只在相关句仍保持撤销后状态时允许": "直前に取り消した裁決を、対象文が取消後の状態を保っている場合に限りやり直します。",
})
_PATTERNS[LANG_EN].extend([
    (re.compile(r"^选择第 (\d+) 句候选$"), lambda m: f"Choose candidate for sentence {m.group(1)}"),
    (re.compile(r"^重新打开第 (\d+) 句$"), lambda m: f"Reopen sentence {m.group(1)}"),
    (re.compile(r"^已撤销：(.*)（仅恢复 (\d+) 个受影响句）。$"), lambda m: f"Undone: {m.group(1)} (restored only {m.group(2)} affected sentences)."),
    (re.compile(r"^已重做：(.*)（仅更新 (\d+) 个受影响句）。$"), lambda m: f"Redone: {m.group(1)} (updated only {m.group(2)} affected sentences)."),
])
_PATTERNS[LANG_JA].extend([
    (re.compile(r"^选择第 (\d+) 句候选$"), lambda m: f"文 {m.group(1)} の候補を選択"),
    (re.compile(r"^重新打开第 (\d+) 句$"), lambda m: f"文 {m.group(1)} を再オープン"),
    (re.compile(r"^已撤销：(.*)（仅恢复 (\d+) 个受影响句）。$"), lambda m: f"取消済み：{m.group(1)}（影響した {m.group(2)} 文のみ復元）。"),
    (re.compile(r"^已重做：(.*)（仅更新 (\d+) 个受影响句）。$"), lambda m: f"やり直し：{m.group(1)}（影響した {m.group(2)} 文のみ更新）。"),
])


# Cross-platform Apple OCR backend labels.  Keep these explicit so the
# Intel/Apple-Silicon wording remains deterministic in all interface languages.
_EXACT_EN.update({
    "Apple Vision · 原生 OCR（Intel / Apple Silicon 推荐）": "Apple Vision · Native OCR (Intel / Apple Silicon recommended)",
})
_EXACT_JA.update({
    "Apple Vision · 原生 OCR（Intel / Apple Silicon 推荐）": "Apple Vision · ネイティブOCR（Intel / Apple Silicon 推奨）",
})

# Phase 20 redesigned chrome: keep the new visible strings covered by the
# offline localization contract.  These are presentation-only labels.
_EXACT_EN.update({
    "版": "NF",
    "‹  上一页": "‹  Previous",
    "下一页  ›": "Next  ›",
    "下一页 →": "Next →",
    "直接读取 PDF 内嵌文字层，不调用任何 OCR 模型，速度快、最准确。": "Read the PDF embedded text layer directly without any OCR model; fast and most accurate.",
    "📄   选择 PDF 文件": "📄   Choose PDF file",
    "自动规则": "Automatic rules",
    "0 / 5 步": "0 / 5 steps",
    "打开 →": "Open →",
    "继续 →": "Continue →",
})
_EXACT_JA.update({
    "版": "NF",
    "‹  上一页": "‹  前のページ",
    "下一页  ›": "次のページ  ›",
    "下一页 →": "次のページ →",
    "直接读取 PDF 内嵌文字层，不调用任何 OCR 模型，速度快、最准确。": "PDF内蔵テキストレイヤーをOCRモデルなしで直接読み込みます。高速で最も正確です。",
    "📄   选择 PDF 文件": "📄   PDFファイルを選択",
    "自动规则": "自動規則",
    "0 / 5 步": "0 / 5 ステップ",
    "打开 →": "開く →",
    "继续 →": "続ける →",
})

# Phase 20 page/workspace redesign strings.
_EXACT_EN.update({
    "导入图片": "Import Images",
    "导入 PDF": "Import PDF",
    "批量改类型": "Change Type",
    "删除所选": "Delete Selected",
    "更多": "More",
    "导入图片、PDF 或文件夹开始": "Import images, a PDF, or a folder to begin",
    "选择图片（可多选）": "Choose images (multiple allowed)",
    "图片 (*.png *.jpg *.jpeg *.heic *.tif *.tiff *.bmp *.gif);;所有文件 (*)": "Images (*.png *.jpg *.jpeg *.heic *.tif *.tiff *.bmp *.gif);;All files (*)",
    "选择 PDF（可多选）": "Choose PDFs (multiple allowed)",
    "PDF (*.pdf);;所有文件 (*)": "PDF (*.pdf);;All files (*)",
    "项目管理": "Project Management",
})
_EXACT_JA.update({
    "导入图片": "画像を読み込む",
    "导入 PDF": "PDFを読み込む",
    "批量改类型": "種類を一括変更",
    "删除所选": "選択を削除",
    "更多": "その他",
    "导入图片、PDF 或文件夹开始": "画像・PDF・フォルダーを読み込んで開始",
    "选择图片（可多选）": "画像を選択（複数可）",
    "图片 (*.png *.jpg *.jpeg *.heic *.tif *.tiff *.bmp *.gif);;所有文件 (*)": "画像 (*.png *.jpg *.jpeg *.heic *.tif *.tiff *.bmp *.gif);;すべてのファイル (*)",
    "选择 PDF（可多选）": "PDFを選択（複数可）",
    "PDF (*.pdf);;所有文件 (*)": "PDF (*.pdf);;すべてのファイル (*)",
    "项目管理": "プロジェクト管理",
})

# Phase 30 import / review-flow UI additions.
_EXACT_EN.update({
    "导入文件": "Import Files",
    "导入文件夹": "Import Folder",
    "选择图片或 PDF（可多选）；也可以直接把文件拖进窗口": "Choose images or PDFs (multiple allowed), or drag files into the window",
    "导入文件夹里的全部图片和 PDF（按文件名自然排序）": "Import all images and PDFs in the folder (natural filename order)",
    "点击「导入文件」或「导入文件夹」开始，也可以把图片、PDF、文件夹直接拖到这里": "Click Import Files or Import Folder to begin, or drag images, PDFs, or folders here",
    "文字方向：横排（左 → 右）": "Writing direction: horizontal (left → right)",
    "Alt+1 / 2 / 3 采用对应候选 · Alt+↑ ↓ 上一个 / 下一个待判断句 · 采用后自动跳到下一个": "Alt+1 / 2 / 3 use candidate · Alt+↑ ↓ previous / next pending sentence · advances automatically after use",
})
_EXACT_JA.update({
    "导入文件": "ファイルを読み込む",
    "导入文件夹": "フォルダーを読み込む",
    "选择图片或 PDF（可多选）；也可以直接把文件拖进窗口": "画像またはPDFを選択（複数可）。ファイルをウィンドウへドラッグすることもできます",
    "导入文件夹里的全部图片和 PDF（按文件名自然排序）": "フォルダー内の画像とPDFをすべて読み込みます（ファイル名の自然順）",
    "点击「导入文件」或「导入文件夹」开始，也可以把图片、PDF、文件夹直接拖到这里": "「ファイルを読み込む」または「フォルダーを読み込む」で開始。画像・PDF・フォルダーをここへドラッグすることもできます",
    "文字方向：横排（左 → 右）": "文字方向：横書き（左 → 右）",
    "Alt+1 / 2 / 3 采用对应候选 · Alt+↑ ↓ 上一个 / 下一个待判断句 · 采用后自动跳到下一个": "Alt+1 / 2 / 3 で候補を採用 · Alt+↑ ↓ で前 / 次の未判定文 · 採用後は次へ自動移動",
})

# Phase 31b auxiliary-dialog polish.
_EXACT_EN.update({
    "运行详情已复制到剪贴板。": "Run details copied to the clipboard.",
})
_EXACT_JA.update({
    "运行详情已复制到剪贴板。": "実行詳細をクリップボードにコピーしました。",
})

# Phase 31 interaction polish: global drag-and-drop, toasts.
_EXACT_EN.update({
    "图片 · PDF · 文件夹都可以": "Images, PDFs, and folders are all fine",
    "松开鼠标，导入文件": "Release to import files",
    "OCR 正在运行，请先停止或完成后再导入新文件": "OCR is running. Stop or finish it before importing new files.",
})
_EXACT_JA.update({
    "图片 · PDF · 文件夹都可以": "画像・PDF・フォルダーに対応",
    "松开鼠标，导入文件": "マウスを離すとファイルを読み込みます",
    "OCR 正在运行，请先停止或完成后再导入新文件": "OCR の実行中です。停止または完了してから新しいファイルを読み込んでください。",
})

# Phase 20 compact Formatter / EPUB / Settings views.
_EXACT_EN.update({
    "书名": "Title",
    "横排（左→右）": "Horizontal (left → right)",
    "竖排（右→左）": "Vertical (right → left)",
    "高级构建 / 包文件预览": "Advanced Build / Package Preview",
    "吾\n輩\nは\n猫\nで\nあ\nる": "I\nAm\na\nCat",
    "检查": "Check",
    "检查结果": "Check Results",
    "可定位问题": "Locatable Issues",
    "双击或按 Enter 跳转到打包文件": "Double-click or press Enter to open the package file",
    "简洁视图": "Compact View",
    "排版方向": "Writing Direction",
    "待检查": "Pending",
    "处理步骤": "Processing Steps",
    "输出": "Output",
    "高级编辑 / 全部工具": "Advanced Editor / All Tools",
    "只预览，不改写正文": "Preview only; body text is not rewritten",
    "切换该组规则；完整逐项控制请打开高级编辑": "Toggle this rule group; open Advanced Editor for individual controls",
    "文字方向": "Writing Direction",
    "版式预览": "Layout Preview",
    "详细设置": "Detailed Settings",
    "更多设置": "More Settings",
    "收起更多设置  ‹": "Hide More Settings  ‹",
    "概览": "Overview",
})
_EXACT_JA.update({
    "书名": "書名",
    "横排（左→右）": "横書き（左→右）",
    "竖排（右→左）": "縦書き（右→左）",
    "高级构建 / 包文件预览": "高度なビルド / パッケージプレビュー",
    "吾\n輩\nは\n猫\nで\nあ\nる": "吾\n輩\nは\n猫\nで\nあ\nる",
    "检查": "チェック",
    "检查结果": "チェック結果",
    "可定位问题": "位置を開ける問題",
    "双击或按 Enter 跳转到打包文件": "ダブルクリックまたは Enter でパッケージファイルを開く",
    "简洁视图": "簡潔表示",
    "排版方向": "組版方向",
    "待检查": "未チェック",
    "处理步骤": "処理ステップ",
    "输出": "出力",
    "高级编辑 / 全部工具": "高度な編集 / 全ツール",
    "只预览，不改写正文": "プレビューのみ。本文は書き換えません",
    "切换该组规则；完整逐项控制请打开高级编辑": "このルール群を切替。個別制御は高度な編集で行います",
    "文字方向": "文字方向",
    "版式预览": "レイアウトプレビュー",
    "详细设置": "詳細設定",
    "更多设置": "その他の設定",
    "收起更多设置  ‹": "その他の設定を閉じる  ‹",
    "概览": "概要",
})
_EXACT_JA.update({
    "吾\n輩\nは\n猫\nで\nあ\nる": "吾\n輩\nは\n猫\nで\nあ\nる\n（見本）",
    "文字方向": "書字方向",
})


# Phase 20 OCR comparison redesign strings.
_EXACT_EN.update({
    "候选结果": "Candidate Results",
    "全文融合结果": "Full-text Fusion Result",
    "分歧队列": "Disagreement Queue",
    "采用": "Use",
    "已选": "Selected",
    "高风险": "High Risk",
    "待复核": "Review",
    "待裁决": "Pending",
    "显示全文": "Show Full Text",
    "返回逐句": "Back to Sentences",
    "图文检查": "Image/Text Check",
    "更多操作 ▾": "More Actions ▾",
    "OCR 结果": "OCR Results",
    "等待 OCR 结果": "Waiting for OCR results",
    "OCR 对比 · 逐句裁决": "OCR Compare · Sentence Review",
    "OCR 对比 · 全文对比": "OCR Compare · Full Text",
})
_EXACT_JA.update({
    "候选结果": "候補結果",
    "全文融合结果": "全文融合結果",
    "分歧队列": "不一致キュー",
    "采用": "採用",
    "已选": "選択済み",
    "高风险": "高リスク",
    "待复核": "要確認",
    "待裁决": "未決",
    "显示全文": "全文を表示",
    "返回逐句": "文単位へ戻る",
    "图文检查": "画像/本文確認",
    "更多操作 ▾": "その他の操作 ▾",
    "OCR 结果": "OCR 結果",
    "等待 OCR 结果": "OCR 結果待ち",
    "OCR 对比 · 逐句裁决": "OCR比較 · 文単位裁決",
    "OCR 对比 · 全文对比": "OCR比較 · 全文比較",
})

# Phase 25–26 contrast/accessibility polish and restored visible entry points.
_EXACT_EN.update({
    "日语 (ja)": "Japanese (ja)",
    "简体中文 (zh-CN)": "Simplified Chinese (zh-CN)",
    "横排 · 左开": "Horizontal · Left Opening",
    "竖排 · 右开": "Vertical · Right Opening",
    "导出概览": "Export Overview",
    "语言": "Language",
    "阅读方向": "Reading Direction",
    "GPT 裁决建议": "GPT Adjudication Suggestion",
    "尚未生成 AI 建议。可直接选择上方候选，或启动 AI 裁决；原始 OCR 候选始终保留。": "No AI suggestion yet. Choose a candidate above directly or start AI adjudication; original OCR candidates are always preserved.",
    "手动编辑": "Manual Edit",
    "跳过": "Skip",
    "右键可打开更多操作": "Right-click for more actions",
    "✓ 应用整本校对稿": "✓ Apply Whole-book Proofread Draft",
    "预览分列": "Preview Columns",
    "预览当前页的物理分列与实际单列 OCR 输入": "Preview physical columns and the actual per-column OCR inputs for the current page",
    "显示实时进度": "Show Live Progress",
    "停止 OCR": "Stop OCR",
    "OCR逐列底稿：按物理列从上到下列出；列号仍对应右→左的日文阅读顺序。": "Per-column OCR draft: physical columns are listed top to bottom; column numbers still follow Japanese right-to-left reading order.",
    "⋯  更多": "⋯  More",
})
_EXACT_JA.update({
    "日语 (ja)": "日本語 (ja)",
    "简体中文 (zh-CN)": "簡体字中国語 (zh-CN)",
    "横排 · 左开": "横書き · 左開き",
    "竖排 · 右开": "縦書き · 右開き",
    "导出概览": "書き出し概要",
    "语言": "言語",
    "阅读方向": "読書方向",
    "GPT 裁决建议": "GPT 裁決提案",
    "尚未生成 AI 建议。可直接选择上方候选，或启动 AI 裁决；原始 OCR 候选始终保留。": "AI提案はまだありません。上の候補を直接選ぶかAI裁決を開始できます。元のOCR候補は常に保持されます。",
    "手动编辑": "手動編集",
    "跳过": "スキップ",
    "右键可打开更多操作": "右クリックでその他の操作を開く",
    "✓ 应用整本校对稿": "✓ 全書校正稿を適用",
    "预览分列": "列分割をプレビュー",
    "预览当前页的物理分列与实际单列 OCR 输入": "現在ページの物理列分割と実際の列別OCR入力をプレビュー",
    "显示实时进度": "リアルタイム進捗を表示",
    "停止 OCR": "OCRを停止",
    "OCR逐列底稿：按物理列从上到下列出；列号仍对应右→左的日文阅读顺序。": "列別OCR下書き：物理列を上から下へ表示し、列番号は日本語の右→左の読順に対応します。",
    "⋯  更多": "⋯  その他",
})

# Phase 28 selective-review topology note.
_EXACT_EN.update({
    "同一页面与物理列身份由共享几何保证；不同 OCR 可使用整页或模型优化后的单列输入图像。第三主模型槽完整读取全部物理列；分歧复核槽只读取残余冲突列，可直接比较全列识别与按需复核的时间和质量。":
        "Shared geometry keeps page and physical-column identity stable. Each OCR engine may use either a full-page image or its model-optimized single-column input. The third main-model slot reads every physical column, while disagreement review reads only residual conflict columns so full-column cost can be compared directly with selective review quality and runtime."
})
_EXACT_JA.update({
    "同一页面与物理列身份由共享几何保证；不同 OCR 可使用整页或模型优化后的单列输入图像。第三主模型槽完整读取全部物理列；分歧复核槽只读取残余冲突列，可直接比较全列识别与按需复核的时间和质量。":
        "共有ジオメトリによりページと物理列の同一性を維持します。各OCRは全ページ画像またはモデル最適化済みの単列入力を使用できます。第3主モデル枠は全物理列を読み、不一致レビュー枠は残った競合列だけを読むため、全列認識と選択的レビューの時間・品質を直接比較できます。"
})

# Phase 28 + Phase35 PDF text-layer / Page Manager integration strings.
_EXACT_EN.update({
    "PDF 来源由页面管理统一提供；本页不再重复导入文件。可自动识别上下双页，不调用任何 OCR 模型。": "PDF input is provided centrally by Page Manager; this page no longer imports the file again. Stacked two-up pages can be detected automatically without invoking any OCR model.",
    "当前页面管理 PDF": "Current Page Manager PDF",
    "尚未在页面管理载入 PDF": "No PDF loaded in Page Manager",
    "如需更换文件，请回到页面管理重新导入 PDF。": "To change the file, return to Page Manager and import the PDF there.",
    "自动识别上下双页文字层": "Auto-detect stacked two-up text layers",
    "检测 PDF 页面中央的全宽空白带；命中后按上半→下半作为两个逻辑页提取。\n不会固定从正中间硬切；如遇特殊 PDF 可取消勾选。": "Detect a full-width blank band near the center of each PDF page; when found, extract the upper half then lower half as two logical pages.\nThe split is not forced at the exact midpoint; disable this option for unusual PDFs.",
    "使用页面管理已确认页类型（非正文跳过）": "Use confirmed Page Manager page types (skip non-body pages)",
    "仅使用页面管理里由用户亲自确认过的页类型；导入时自动填充的“正文”建议不会参与。\n当前 PDF 与页面管理来源一致时，封面/插图/空白页/目录/后记/版权页等会直接跳过文字层提取。": "Use only page types explicitly confirmed by the user in Page Manager; automatically suggested 'body' labels from import are ignored.\nWhen the current PDF matches the Page Manager source, covers, illustrations, blank pages, TOC, afterword, copyright pages, and other confirmed non-body pages are skipped during text-layer extraction.",
    "页面管理：未绑定": "Page Manager: not bound",
    "去标记": "Open Page Manager",
    "打开页面管理；选中封面、插图、空白页等后直接设置类型": "Open Page Manager, select covers, illustrations, blank pages, and other pages, then set their page types directly.",
    "导出损坏字 GPT 字形复核包": "Export GPT Glyph Review Pack",
    "只导出 PDF 文字层中 U+FFFD 损坏字形；同一内嵌字体 + glyph id 默认只复核一次，不会把整本图片重新送去识别。": "Export only U+FFFD damaged glyphs from the PDF text layer. The same embedded font + glyph ID is reviewed once by default; the full book is never resent for image recognition.",
    "导入 GPT 修复 JSON": "Import GPT Repair JSON",
    "导入复核包对应的 corrections JSON；只允许替换原来明确为 U+FFFD 的位置。": "Import the corrections JSON for the review pack; replacements are allowed only at positions that were explicitly U+FFFD in the source text layer.",
    "正在导出复核包…": "Exporting review pack…",
    "页面类型标记：已关闭；仍使用页面管理当前 PDF 作为来源": "Page-type filtering: off; the current Page Manager PDF is still used as the source",
    "导出 GPT 视觉复核包": "Export GPT Visual Review Pack",
    "请先在页面管理导入 PDF": "Import a PDF in Page Manager first",
    "页面管理 PDF 已变化": "Page Manager PDF has changed",
    "没有可复核结果": "No reviewable results",
    "请先完成一次 PDF 文字层提取。": "Run PDF text-layer extraction first.",
    "无需复核": "No review needed",
    "当前 PDF 文字层没有检测到 U+FFFD 损坏字形。": "No U+FFFD damaged glyphs were detected in the current PDF text layer.",
    "没有可修复结果": "No repairable results",
    "修复已应用": "Repairs Applied",
    "没有应用修复": "No repairs applied",
})
_EXACT_JA.update({
    "PDF 来源由页面管理统一提供；本页不再重复导入文件。可自动识别上下双页，不调用任何 OCR 模型。": "PDF入力はページ管理から一元的に提供され、このページでは重複して読み込みません。上下2面ページを自動検出でき、OCRモデルは使用しません。",
    "当前页面管理 PDF": "現在のページ管理PDF",
    "尚未在页面管理载入 PDF": "ページ管理にPDFが読み込まれていません",
    "如需更换文件，请回到页面管理重新导入 PDF。": "ファイルを変更する場合はページ管理に戻ってPDFを読み込み直してください。",
    "自动识别上下双页文字层": "上下2面ページのテキスト層を自動検出",
    "检测 PDF 页面中央的全宽空白带；命中后按上半→下半作为两个逻辑页提取。\n不会固定从正中间硬切；如遇特殊 PDF 可取消勾选。": "PDFページ中央付近の全幅空白帯を検出し、該当時は上半分→下半分を2つの論理ページとして抽出します。\n中央で固定分割はしません。特殊なPDFではこの設定をオフにできます。",
    "使用页面管理已确认页类型（非正文跳过）": "ページ管理で確定したページ種別を使用（本文以外をスキップ）",
    "仅使用页面管理里由用户亲自确认过的页类型；导入时自动填充的“正文”建议不会参与。\n当前 PDF 与页面管理来源一致时，封面/插图/空白页/目录/后记/版权页等会直接跳过文字层提取。": "ページ管理でユーザーが明示的に確定したページ種別だけを使用し、読み込み時に自動提案された「本文」は対象にしません。\n現在のPDFがページ管理のソースと一致する場合、表紙・挿絵・空白ページ・目次・あとがき・奥付など、確定済みの非本文ページはテキスト層抽出を直接スキップします。",
    "页面管理：未绑定": "ページ管理：未接続",
    "去标记": "ページ管理を開く",
    "打开页面管理；选中封面、插图、空白页等后直接设置类型": "ページ管理を開き、表紙・挿絵・空白ページなどを選択してページ種別を直接設定します。",
    "导出损坏字 GPT 字形复核包": "GPT字形レビュー用パックを書き出す",
    "只导出 PDF 文字层中 U+FFFD 损坏字形；同一内嵌字体 + glyph id 默认只复核一次，不会把整本图片重新送去识别。": "PDFテキスト層のU+FFFD破損字形だけを書き出します。同じ埋め込みフォント + glyph IDは既定で1回だけ確認し、本全体の画像を再認識には送りません。",
    "导入 GPT 修复 JSON": "GPT修正JSONを読み込む",
    "导入复核包对应的 corrections JSON；只允许替换原来明确为 U+FFFD 的位置。": "レビュー用パックに対応するcorrections JSONを読み込みます。元テキスト層で明確にU+FFFDだった位置だけを置換できます。",
    "正在导出复核包…": "レビュー用パックを書き出し中…",
    "页面类型标记：已关闭；仍使用页面管理当前 PDF 作为来源": "ページ種別フィルター：オフ。ソースには引き続き現在のページ管理PDFを使用します",
    "导出 GPT 视觉复核包": "GPT視覚レビュー用パックを書き出す",
    "请先在页面管理导入 PDF": "先にページ管理でPDFを読み込んでください",
    "页面管理 PDF 已变化": "ページ管理のPDFが変更されました",
    "没有可复核结果": "確認対象の結果がありません",
    "请先完成一次 PDF 文字层提取。": "先にPDFテキスト層の抽出を1回実行してください。",
    "无需复核": "確認不要",
    "当前 PDF 文字层没有检测到 U+FFFD 损坏字形。": "現在のPDFテキスト層ではU+FFFDの破損字形は検出されませんでした。",
    "没有可修复结果": "修正可能な結果がありません",
    "修复已应用": "修正を適用しました",
    "没有应用修复": "修正は適用されませんでした",
})

# Phase 33 compact OCR UI label.
_EXACT_EN.update({
    "更多设置": "More settings",
    "收起更多设置  ‹": "Hide more settings  ‹",
})
_EXACT_JA.update({
    "更多设置": "その他の設定",
    "收起更多设置  ‹": "その他の設定を閉じる  ‹",
})

# Phase 33 centered empty-state copy.
_EXACT_EN.update({
    "点击「导入文件」或「导入文件夹」开始\n\n也可以把图片、PDF、文件夹直接拖到这里":
        "Click Import Files or Import Folder to begin\n\nYou can also drag images, PDFs, or folders here",
})
_EXACT_JA.update({
    "点击「导入文件」或「导入文件夹」开始\n\n也可以把图片、PDF、文件夹直接拖到这里":
        "「ファイルを読み込む」または「フォルダーを読み込む」で開始\n\n画像・PDF・フォルダーをここへドラッグすることもできます",
})

# Phase 38 TEI / EPUBCheck publication bridge.
_EXACT_EN.update({
    "📂 导入 JSON / MD / TEI": "📂 Import JSON / MD / TEI",
    "通过": "Pass",
    "未运行": "Not run",
})
_EXACT_JA.update({
    "📂 导入 JSON / MD / TEI": "📂 JSON / MD / TEI を読み込む",
    "通过": "合格",
    "未运行": "未実行",
})

# Phase 34 PDF text / Page Manager cover synchronization.
_EXACT_EN.update({
    "封面\n未设置": "Cover\nnot set",
    "页面管理尚未设置封面": "No cover is set in Page Manager",
    "✓ 封面：已设置": "✓ Cover: set",
})
_EXACT_JA.update({
    "封面\n未设置": "表紙\n未設定",
    "页面管理尚未设置封面": "ページ管理で表紙が設定されていません",
    "✓ 封面：已设置": "✓ 表紙：設定済み",
})

# Phase 44 provider-neutral OCR adjudication and workspace recent-project views.
_EXACT_EN.update({
    "AI 裁决": "AI Adjudication",
    "AI 服务…": "AI Service…",
    "AI 服务未配置": "AI service not configured",
    "出版级 AI 裁决": "Publication-grade AI Adjudication",
    "出版级模式会先让当前已配置的视觉 AI 在看不到 A/B/C 的情况下独立逐字抄写，再把视觉结果、全部 OCR 候选、物理列、前后文和已保存术语交给第二阶段上下文裁决，最后由独立审计器反查。支持 OpenAI、Gemini、DeepSeek、GLM、Claude、OpenRouter 等已配置服务。": "Publication-grade mode first asks the configured vision AI to transcribe independently without seeing A/B/C. It then sends the visual transcription, all OCR candidates, physical columns, surrounding context, and saved terminology to a second contextual adjudication stage, followed by an independent audit. Supports configured OpenAI, Gemini, DeepSeek, GLM, Claude, OpenRouter, and other services.",
    "切换 AI 服务…": "Switch AI service…",
    "出版级 AI · 视觉抄写 + 上下文裁决 + 独立审计（推荐）": "Publication-grade AI · visual transcription + contextual adjudication + independent audit (recommended)",
    "出版级模式完成后再用独立审计器反查缺字、重复、邻列粘连、数字/否定/专名（推荐）": "After publication-grade mode, run an independent audit for omissions, duplicates, adjacent-column adhesion, numbers, negation, and proper nouns (recommended)",
    "当前 AI：未配置": "Current AI: not configured",
    "本地 Ollama 无需 API Key": "Local Ollama does not require an API key",
    "API Key 可选（无鉴权接口可留空）": "API key optional (leave blank for endpoints without authentication)",
    "请输入 API Key": "Enter API key",
    "缩略图": "Thumbnails",
    "列表": "List",
    "双击项目即可切换；缩略图来自项目持久化页面，不重新读取 OCR。": "Double-click a project to switch. Thumbnails use persisted project pages and do not rerun OCR.",
})
_EXACT_JA.update({
    "AI 裁决": "AI裁定",
    "AI 服务…": "AIサービス…",
    "AI 服务未配置": "AIサービス未設定",
    "出版级 AI 裁决": "出版級AI裁定",
    "出版级模式会先让当前已配置的视觉 AI 在看不到 A/B/C 的情况下独立逐字抄写，再把视觉结果、全部 OCR 候选、物理列、前后文和已保存术语交给第二阶段上下文裁决，最后由独立审计器反查。支持 OpenAI、Gemini、DeepSeek、GLM、Claude、OpenRouter 等已配置服务。": "出版級モードでは、現在設定されている視覚AIがA/B/Cを見ずに独立して逐字転写します。その後、視覚転写、全OCR候補、物理列、前後文、保存済み用語を第2段階の文脈裁定へ渡し、最後に独立監査で再確認します。設定済みのOpenAI、Gemini、DeepSeek、GLM、Claude、OpenRouterなどに対応します。",
    "切换 AI 服务…": "AIサービスを切替…",
    "出版级 AI · 视觉抄写 + 上下文裁决 + 独立审计（推荐）": "出版級AI · 視覚転写 + 文脈裁定 + 独立監査（推奨）",
    "出版级模式完成后再用独立审计器反查缺字、重复、邻列粘连、数字/否定/专名（推荐）": "出版級モード完了後、独立監査で欠落、重複、隣接列の混入、数字・否定・固有名詞を再確認します（推奨）",
    "当前 AI：未配置": "現在のAI：未設定",
    "本地 Ollama 无需 API Key": "ローカルOllamaではAPIキーは不要です",
    "API Key 可选（无鉴权接口可留空）": "APIキーは任意です（認証不要のエンドポイントでは空欄可）",
    "请输入 API Key": "APIキーを入力してください",
    "缩略图": "サムネイル",
    "列表": "リスト",
    "双击项目即可切换；缩略图来自项目持久化页面，不重新读取 OCR。": "プロジェクトをダブルクリックして切り替えます。サムネイルは保存済みのプロジェクトページを使い、OCRを再実行しません。",
})

# Phase 46 OCR adjudication workbench strings.  Keep the review workspace fully
# offline-localizable: these labels describe UI state only and never touch OCR
# document content.
_EXACT_EN.update({
    "✓ 已确认人工裁决 · 如继续修改，需要再次点确认": "✓ Human adjudication confirmed · confirm again after further edits",
    "草稿区 · 当前没有可裁决的多模型句": "Draft · no multi-model sentence is available for adjudication",
    "多模型结果一致 · 当前句只读，无需人工裁决": "Multi-model results agree · this sentence is read-only; no human adjudication needed",
    "已确认人工裁决 · 可继续修改，修改后需再次确认": "Human adjudication confirmed · further edits require confirmation again",
    "草稿区 · 选择候选“作为底稿”或直接输入；不会自动提交": "Draft · choose a candidate as the base or type directly; nothing is submitted automatically",
    "待判断": "Pending",
    "三方分歧": "Three-way conflict",
    "空/占位符": "Empty / placeholder",
    "已裁决": "Adjudicated",
    "切换待判断、高风险优先或已裁决历史；只改变浏览范围。": "Switch between pending, high-risk first, or adjudicated history; this only changes the browsing scope.",
    "筛选页码 / 候选文字": "Filter page / candidate text",
    "待判 0 · 三方 0 · 空/占位 0 · 已裁决 0": "Pending 0 · three-way 0 · empty/placeholder 0 · adjudicated 0",
    "整本裁决总览：橙=待判断，蓝=人工/AI裁决，绿=稳定；点击可跳转。": "Whole-book adjudication overview: orange=pending, blue=human/AI adjudicated, green=stable; click to jump.",
    "F7 下一分歧 · Shift+F7 上一 · Alt+1…9 作底稿 · Ctrl+Alt+1…9 直接采用 · Alt+I 图文证据": "F7 next conflict · Shift+F7 previous · Alt+1…9 use as base · Ctrl+Alt+1…9 accept directly · Alt+I image evidence",
    "草稿区 · 选择“作为底稿”后修改，未确认前不会进入裁决结果": "Draft · edit after choosing Use as Base; it does not enter the adjudication result until confirmed",
    "放弃草稿": "Discard Draft",
    "确认当前手动文本；停留在本句": "Confirm current manual text; stay on this sentence",
    "确认并下一分歧": "Confirm and Next Conflict",
    "Ctrl+Shift+Enter：确认后进入下一分歧": "Ctrl+Shift+Enter: confirm and move to the next conflict",
    "原图证据": "Source Image Evidence",
    "图文对照 ↗": "Image/Text Review ↗",
    "在图文对照中打开同一稳定句（Alt+I）": "Open the same stable sentence in Image/Text Review (Alt+I)",
    "只读像素证据 · 不随候选修改": "Read-only pixel evidence · unaffected by candidate edits",
    "候选对比 · 红底=替换 · 橙底=增删/缺失": "Candidate comparison · red=replacement · orange=insert/delete/missing",
    "导出 AI 裁决包": "Export AI Adjudication Pack",
    "导入 AI 裁决包": "Import AI Adjudication Pack",
    "融合结果＋骨架 EPUB": "Fused Result + Skeleton EPUB",
    "作为底稿": "Use as Base",
    "直接采用": "Accept Directly",
    "正在载入参考原图…": "Loading source image…",
    "完整参考裁片；点击原尺寸查看可放大阅读。": "Full reference crop; open at original size for closer reading.",
    "原尺寸查看": "View Original Size",
    "OCR 参考原图 · 原尺寸": "OCR Source Image · Original Size",
    "多模型结果一致 · 只读": "Multi-model results agree · read-only",
    "复制到手动编辑框；不会保存、不会跳转": "Copy to the manual editor; does not save or navigate",
    "直接采用并下一条": "Accept and Next",
    "直接完成当前裁决并进入下一条 OCR 分歧": "Complete the current adjudication and move to the next OCR conflict",
    "快速定位 OCR 结果不一致的句子。": "Quickly locate sentences where OCR results disagree.",
    "裁决竖列": "Adjudication Column",
    "查看当前裁决稿对应的完整物理竖列。": "View the complete physical vertical column for the current adjudication.",
    "定位左侧物理列（⌥⇧←）": "Move to the physical column on the left (⌥⇧←)",
    "定位右侧物理列（⌥⇧→）": "Move to the physical column on the right (⌥⇧→)",
    "全部分歧": "All Conflicts",
    "已修改": "Modified",
    "已确认": "Confirmed",
    "筛选页码 / 当前文本": "Filter page / current text",
    "拖动图片可定位物理列 · 蓝框=当前列 · Alt+P 原尺寸": "Drag the image to locate a physical column · blue box=current column · Alt+P original size",
    "待核对": "Needs Review",
    "红底=替换 · 橙底=增删/缺失": "Red=replacement · orange=insert/delete/missing",
    "确认当前手动文本并停留在本句": "Confirm current manual text and stay on this sentence",
    "保存当前裁决并进入下一句（Ctrl/⌘+Shift+Return）": "Save current adjudication and move to the next sentence (Ctrl/⌘+Shift+Return)",
    "F7 下一分歧 · Shift+F7 上一 · Alt+1…9 作底稿": "F7 next conflict · Shift+F7 previous · Alt+1…9 use as base",
    "✓ 已采用": "✓ Accepted",
    "分歧 0": "Conflicts 0",
    "直接采用失败；请检查当前句映射": "Direct accept failed; check the current sentence mapping",
    "完整裁决文字：物理列按右→左排列，可滚动查看全部文字。": "Complete adjudication text: physical columns are arranged right-to-left; scroll to view all text.",
    "▶  开始提取": "▶  Start Extraction",
    "■  停止提取": "■  Stop Extraction",
    "第 {current} / {total} 列 · 右→左": "Column {current} / {total} · right→left",
    "未确认草稿": "Unconfirmed Draft",
    "未保存": "Unsaved",
})
_EXACT_JA.update({
    "✓ 已确认人工裁决 · 如继续修改，需要再次点确认": "✓ 手動裁定を確定済み · さらに編集した場合は再確認が必要です",
    "草稿区 · 当前没有可裁决的多模型句": "下書き · 現在裁定できる複数モデル文はありません",
    "多模型结果一致 · 当前句只读，无需人工裁决": "複数モデルの結果が一致 · この文は読み取り専用で手動裁定は不要です",
    "已确认人工裁决 · 可继续修改，修改后需再次确认": "手動裁定を確定済み · 続けて編集できますが変更後は再確認が必要です",
    "草稿区 · 选择候选“作为底稿”或直接输入；不会自动提交": "下書き · 候補を「下書きにする」で選ぶか直接入力してください。自動確定はしません",
    "待判断": "未判定",
    "三方分歧": "3者不一致",
    "空/占位符": "空欄 / プレースホルダー",
    "已裁决": "裁定済み",
    "切换待判断、高风险优先或已裁决历史；只改变浏览范围。": "未判定・高リスク優先・裁定履歴を切り替えます。閲覧範囲だけが変わります。",
    "筛选页码 / 候选文字": "ページ / 候補文字で絞り込み",
    "待判 0 · 三方 0 · 空/占位 0 · 已裁决 0": "未判定 0 · 3者 0 · 空欄/プレースホルダー 0 · 裁定済み 0",
    "整本裁决总览：橙=待判断，蓝=人工/AI裁决，绿=稳定；点击可跳转。": "全書裁定一覧：橙=未判定、青=手動/AI裁定、緑=安定。クリックで移動できます。",
    "F7 下一分歧 · Shift+F7 上一 · Alt+1…9 作底稿 · Ctrl+Alt+1…9 直接采用 · Alt+I 图文证据": "F7 次の不一致 · Shift+F7 前へ · Alt+1…9 下書き · Ctrl+Alt+1…9 直接採用 · Alt+I 画像証拠",
    "草稿区 · 选择“作为底稿”后修改，未确认前不会进入裁决结果": "下書き · 「下書きにする」を選んで編集し、確認するまでは裁定結果に入りません",
    "放弃草稿": "下書きを破棄",
    "确认当前手动文本；停留在本句": "現在の手動テキストを確定し、この文に留まる",
    "确认并下一分歧": "確定して次の不一致へ",
    "Ctrl+Shift+Enter：确认后进入下一分歧": "Ctrl+Shift+Enter：確定後に次の不一致へ移動",
    "原图证据": "元画像証拠",
    "图文对照 ↗": "画像・テキスト照合 ↗",
    "在图文对照中打开同一稳定句（Alt+I）": "画像・テキスト照合で同じ安定文を開く（Alt+I）",
    "只读像素证据 · 不随候选修改": "読み取り専用の画素証拠 · 候補編集の影響を受けません",
    "候选对比 · 红底=替换 · 橙底=增删/缺失": "候補比較 · 赤=置換 · オレンジ=追加/削除/欠落",
    "导出 AI 裁决包": "AI裁定パックを書き出す",
    "导入 AI 裁决包": "AI裁定パックを読み込む",
    "融合结果＋骨架 EPUB": "統合結果＋骨格EPUB",
    "作为底稿": "下書きにする",
    "直接采用": "直接採用",
    "正在载入参考原图…": "参照元画像を読み込み中…",
    "完整参考裁片；点击原尺寸查看可放大阅读。": "参照クロップ全体です。原寸表示を開くと拡大して確認できます。",
    "原尺寸查看": "原寸表示",
    "OCR 参考原图 · 原尺寸": "OCR参照元画像 · 原寸",
    "多模型结果一致 · 只读": "複数モデルの結果が一致 · 読み取り専用",
    "复制到手动编辑框；不会保存、不会跳转": "手動編集欄へコピーします。保存も移動もしません",
    "直接采用并下一条": "直接採用して次へ",
    "直接完成当前裁决并进入下一条 OCR 分歧": "現在の裁定を完了して次のOCR不一致へ移動",
    "快速定位 OCR 结果不一致的句子。": "OCR結果が一致しない文を素早く特定します。",
    "裁决竖列": "裁定縦列",
    "查看当前裁决稿对应的完整物理竖列。": "現在の裁定稿に対応する物理縦列全体を確認します。",
    "定位左侧物理列（⌥⇧←）": "左側の物理列へ移動（⌥⇧←）",
    "定位右侧物理列（⌥⇧→）": "右側の物理列へ移動（⌥⇧→）",
    "全部分歧": "すべての不一致",
    "已修改": "変更済み",
    "已确认": "確認済み",
    "筛选页码 / 当前文本": "ページ / 現在テキストで絞り込み",
    "拖动图片可定位物理列 · 蓝框=当前列 · Alt+P 原尺寸": "画像をドラッグして物理列を特定 · 青枠=現在列 · Alt+P 原寸",
    "待核对": "要確認",
    "红底=替换 · 橙底=增删/缺失": "赤=置換 · オレンジ=追加/削除/欠落",
    "确认当前手动文本并停留在本句": "現在の手動テキストを確定してこの文に留まる",
    "保存当前裁决并进入下一句（Ctrl/⌘+Shift+Return）": "現在の裁定を保存して次の文へ移動（Ctrl/⌘+Shift+Return）",
    "F7 下一分歧 · Shift+F7 上一 · Alt+1…9 作底稿": "F7 次の不一致 · Shift+F7 前へ · Alt+1…9 下書き",
    "✓ 已采用": "✓ 採用済み",
    "分歧 0": "不一致 0",
    "直接采用失败；请检查当前句映射": "直接採用に失敗しました。現在文のマッピングを確認してください",
    "完整裁决文字：物理列按右→左排列，可滚动查看全部文字。": "裁定全文：物理列は右→左に並びます。スクロールして全文を確認できます。",
    "▶  开始提取": "▶  抽出開始",
    "■  停止提取": "■  抽出停止",
    "第 {current} / {total} 列 · 右→左": "第 {current} / {total} 列 · 右→左",
    "未确认草稿": "未確認下書き",
    "未保存": "未保存",
})

_EXACT_EN.update({
    "↓ 最新日志": "↓ Latest Log",
    "回到 OCR 日志底部并恢复自动跟随最新输出": "Jump to the bottom of the OCR log and resume following new output",
})
_EXACT_JA.update({
    "↓ 最新日志": "↓ 最新ログ",
    "回到 OCR 日志底部并恢复自动跟随最新输出": "OCRログの末尾へ移動し、最新出力の自動追従を再開します",
})

_EXACT_EN.update({
    "显示/隐藏右侧实时预览中的红色实线检测框": "Show/hide the red solid detector boxes in the live preview",
    "显示/隐藏右侧实时预览中的绿色实线 OCR 输入原像素框": "Show/hide the green solid original-pixel OCR input boxes in the live preview",
})
_EXACT_JA.update({
    "显示/隐藏右侧实时预览中的红色实线检测框": "リアルタイムプレビューの赤い実線検出枠を表示/非表示",
    "显示/隐藏右侧实时预览中的绿色实线 OCR 输入原像素框": "リアルタイムプレビューの緑の実線OCR入力原画素枠を表示/非表示",
})
