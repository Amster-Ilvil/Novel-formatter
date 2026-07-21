# Novel Formatter Studio — Python 后端

## 项目结构

```
novel_formatter/
├── run.py                          ← 一键运行入口
│
├── models/
│   └── document.py                 ← 统一文档模型（所有模块共用的数据结构）
│
├── adapters/
│   └── apple_vision_adapter.py     ← Apple Vision OCR 适配器（调用 macOS 快捷指令）
│
├── engine/
│   └── formatter.py                ← Novel Formatter Engine（六步处理流水线）
│
└── builder/
    └── epub_builder.py             ← EPUB3 Builder（生成最终电子书）
```

## 快速开始

### 前置条件
1. macOS Monterey (12) 或更高版本
2. 打开「快捷指令」App，新建快捷指令，命名为 `ExtractText`
3. 添加动作「从图像中提取文字」，输入设为「快捷指令输入」
4. 测试：`shortcuts run ExtractText -i /任意图片.jpg`

### 安装依赖
```bash
pip install python-docx   # 如需同时输出 docx
# epub_builder 使用 Python 标准库 zipfile，无需额外安装
```

### 最简用法

```bash
# 图片文件夹 → EPUB（一键）
python run.py /path/to/novel_images/ 魔法科高校_01.epub \
    --title "魔法科高校の劣等生 第1巻" \
    --author "佐島勤"
```

### 带手动标注

```bash
# 1. 先跑 OCR，保存中间 JSON
python run.py /path/to/images/ output.epub --save-json ocr_result.json

# 2. 用 Page Manager UI 检查页面分类，导出 overrides.json：
#    {"1": "cover", "2": "color_illus", "5": "blank", "28": "illustration"}

# 3. 用标注覆盖重跑（跳过 OCR）
python run.py --from-json ocr_result.json output.epub --overrides overrides.json
```

### 只跑 Formatter（已有 JSON）

```bash
python run.py --from-json ocr_result.json output.epub \
    --steps merge_sentences dialogue_restore detect_chapters normalize_punct
```

---

## 各模块详解

### 统一文档模型 (`models/document.py`)

所有模块只操作 `UnifiedDocument`，不依赖具体文件格式。

```python
from models.document import UnifiedDocument, Block, BlockType

doc = UnifiedDocument.from_json(open("result.json").read())
for block in doc.text_blocks():
    print(block.type, block.text[:30])
```

**BlockType 枚举：**

| 值 | 含义 |
|---|---|
| `cover` | 封面图片页 |
| `color_illus` | 彩色插图页 |
| `blank` | 空白页 |
| `toc_page` | 目录页（原始图） |
| `illustration` | 黑白插图页 |
| `paragraph` | 正文段落 |
| `dialogue` | 对白「」 |
| `chapter` | 章节标题 |
| `image_ref` | 图片引用（内联于正文） |

---

### Apple Vision 适配器 (`adapters/apple_vision_adapter.py`)

调用 macOS 快捷指令，逐页 OCR，输出 `UnifiedDocument`。

**核心逻辑（继承自 `ocr_via_shortcuts.py`）：**
- `detect_running_headers()`：统计每行在多少页出现，超过阈值 → 视为页眉过滤
- `classify_page()`：字数+标点数判定 → 正文 / 插图 / 空白
- `lines_to_blocks()`：每行文字 → 对应 BlockType（章节/对白/段落）

**Page Manager 标注覆盖：**
```python
from adapters.apple_vision_adapter import run

doc = run(
    image_folder="/images",
    page_overrides={1: "cover", 5: "blank", 28: "illustration"},
)
```

---

### Formatter Engine (`engine/formatter.py`)

六个独立步骤，每步接收并返回新的 `UnifiedDocument`（深拷贝，不修改原始数据）。

| 步骤 ID | 函数 | 作用 |
|---|---|---|
| `clean_metadata` | `clean_metadata_blocks` | 删除残留页眉/页码块 |
| `merge_sentences` | `merge_broken_sentences` | 合并竖排 OCR 断句 |
| `remove_duplicates` | `remove_duplicates` | 删除重复段落/对白 |
| `dialogue_restore` | `restore_dialogue_breaks` | 对白独立换行 |
| `detect_chapters` | `detect_chapters` | 章节识别 + TOC 生成 |
| `normalize_punct` | `normalize_punctuation` | 标点规范 + 常见错字修正 |

```python
from engine.formatter import run_pipeline
from models.document import UnifiedDocument

doc = UnifiedDocument.from_json(open("ocr.json").read())

# 全部步骤
result = run_pipeline(doc)

# 只跑部分步骤
result = run_pipeline(doc, steps=["merge_sentences", "detect_chapters"])
```

---

### EPUB Builder (`builder/epub_builder.py`)

从 `UnifiedDocument` 生成符合 EPUB3 规范的 `.epub` 文件。

**CSS 模板：**
- `denki`（默认）：電撃文庫风格，竖排，明朝体
- `mf`：MF文庫J风格，竖排，稍小字号
- `web`：横排，适合 Web 小说

**插图位置恢复：**
每个 `IMAGE_REF` 块带有 `image_anchor` 字段（如 `block_42`），EPUB Builder
在生成对应章节时，在该段落后插入独立的插图 xhtml 页，而非把所有图片放到最后。

```python
from builder.epub_builder import build_epub

build_epub(
    doc,
    output_path="output.epub",
    css_template="denki",
    vertical=True,
)
```

---

## Page Manager 标注格式

`overrides.json` 格式（由 Page Manager UI 导出）：

```json
{
  "1": "cover",
  "2": "color_illus",
  "3": "color_illus",
  "5": "blank",
  "28": "illustration",
  "215": "colophon"
}
```

键为页码（1-based 字符串），值为 `BlockType` 字符串。

---

## 与原脚本 `ocr_via_shortcuts.py` 的关系

| 原脚本 | 本项目 |
|---|---|
| `extract_text()` | `apple_vision_adapter.extract_text_via_shortcut()` — 完全保留 |
| `find_running_headers()` | `detect_running_headers()` — 算法相同，阈值改为比例参数 |
| `filter_lines()` | `filter_lines()` — 完全保留 |
| `is_body_text()` | `classify_page()` — 扩展为返回 BlockType + 置信度 |
| 直接写 docx | 写入 `UnifiedDocument` → 可输出 JSON / EPUB / docx |
| 无 Page Manager | `page_overrides` 参数覆盖自动分类 |
