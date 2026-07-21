# Novel Formatter Studio
## Software Design Specification（SDS）

版本 1.0　2026-07-18

---

# 第一章　项目目标

Novel Formatter Studio **不是 OCR 软件**，而是一款面向日文竖排小说（轻小说 / 漫画文字层）的
**Document Reconstruction Engine**（文档重建引擎），最终产出出版级 EPUB3。

```
任意输入（PDF / 图片 / OCR JSON / Word / …）
        │
        ▼
Document Reconstruction Engine
        │
        ▼
出版级 EPUB
```

OCR 只是众多输入方式之一，核心能力是把碎片化的文字/图像信息**重建**成一本结构完整、
排版正确、可发布的书。所有设计决策都应服务于这个目标，而不是服务于"识别准确率"。

---

# 第二章　整体架构

```
Importer
    │
    ▼
Unified Document（Book）
    │
    ▼
Page Manager（页面分类，人工可修正）
    │
    ▼
OCR Adapter（可选，仅正文/目录页需要）
    │
    ▼
Document Reconstruction Engine
    │
    ├── Page Classification      已在 Page Manager 完成，此处二次确认
    ├── Layout Analysis          Block → Column → Paragraph
    ├── Reading Order            竖排：右→左，列内上→下
    ├── Paragraph Recovery       断句/断行/缩进恢复
    ├── Dialogue Recovery        「」『』独立成行
    ├── Image Reconstruction     插图 anchor 定位
    ├── Chapter Reconstruction   章节 + TOC
    ├── Metadata Reconstruction  封面/目录/OCR/AI 来源合并
    └── Style Reconstruction     竖排/Ruby/CSS
    │
    ▼
AI Enhancement（可选，Diff + Accept/Reject）
    │
    ▼
EPUB Builder
```

**贯穿全文的核心原则**（详见第十三章）：

> 任何模块都不应该直接修改原始数据，而是基于统一的 `Book`（`UnifiedDocument`）
> 数据模型产生新的不可变版本。

```
Raw OCR → Book V1 → Book V2(分类) → Book V3(阅读顺序) → Book V4(段落恢复)
        → Book V5(AI校对) → Book V6(EPUB导出)
```

这是本项目当前实现中**已经部分做到、但未完全暴露**的机制——Formatter 的每一步都对输入
`deepcopy` 后返回新对象，只是版本链没有保留、GUI 也不支持 Undo/Redo。第十三章会给出
具体的补齐方案。

---

# 第三章　Unified Document（Book）

这是整个项目最重要的数据结构。所有模块（Page Manager / OCR / Formatter / AI / EPUB）
只读写这一个结构，不直接依赖具体文件格式。

```
Book
├── Metadata          title / author / publisher / series / volume / language / isbn
├── Pages[]           page_no / page_type / image_path / confidence
├── Blocks[]          type / text / page / bbox / reading_order / confidence /
│                     ocr_raw / modified_by / image_path / image_anchor / chapter_index
├── TOC[]             title / chapter_index / block_index
├── ProcessingLog[]   step / message / count
└── History[]         （新增）每个处理步骤的不可变快照，见第十三章
```

**Block 类型（BlockType）现状与扩展**：

页面级（Page Manager 产出）：
```
cover / color_illus / blank / toc_page / illustration /
afterword / colophon / unknown
```

审阅反馈建议扩展的页面级类型（第十三章已实现）：
```
half_illustration   半页插图
title_page          扉页 / 书名页
frontispiece        卷首插画
insert              插页
advertisement       广告页
index               索引
appendix            附录
map                 地图页
character_sheet     人物设定页
```

文本块级（Formatter 产出）：
```
paragraph / dialogue / chapter / section / ruby /
toc_entry / image_ref / header_footer / footnote（新增）
```

不写完整代码实现，字段定义已足够用于补全具体逻辑。

---

# 第四章　Import Adapter

统一转换入口。目标格式全部转为 `Book`：

```
支持输入:
  PDF（扫描/文本）  Image（png/jpg/heic/tiff）  Word（docx）
  Markdown  TXT  HTML  JSON  OCR JSON  EPUB（逆向导入）
```

当前已实现：图片文件夹 / 单图 / 多图 / PDF（经 `pdf_input.py` 转图片）。
未实现：docx / html / md / txt / epub 逆向导入 —— 留待第十二章 Roadmap 第二阶段。

---

# 第五章　OCR Adapter

插件化接口，不关心具体 OCR 引擎实现：

```
class OCRProvider:
    def detect(self, path) -> bool          # 能否处理该输入
    def recognize(self, image) -> RawResult # 执行识别
    def supports_vertical(self) -> bool     # 是否原生支持竖排
    def supports_bbox(self) -> bool         # 是否返回坐标框
```

当前已实现：`ApplevisionAdapter`（通过 macOS 快捷指令调用 Live Text）。
接口预留：`pdf-craft` / `PaddleOCR` / `Google Vision` —— GUI 中已展示为"即将支持"占位卡片。

新引擎接入只需实现上述接口并注册到 `OCR_ADAPTERS` 列表，不需要改动
Page Manager / Formatter / EPUB Builder 任何一行代码——这正是 Import/OCR 与
Document Reconstruction 解耦的意义所在。

---

# 第六章　Document Reconstruction（核心）

## Module 01　Page Classification
**输入**：Page Image　**输出**：PageType

```
第一页 + 图片占比 >90% + 文字很少   → Cover
文字数 <5                          → Blank
文字数 <30 或 标点数 <5             → Illustration
```
规则优先，AI 只处理规则判定为"低置信度"的页面。

## Module 02　Layout Analysis
```
OCR Blocks → Column → Paragraph → Layout Tree
```
采用 Bounding Box 做版面切分，不引入版面分析论文级别的复杂度。

## Module 03　Reading Order（重点）
**输入**：OCR Block（带 bbox）　**输出**：Paragraph Sequence

```
竖排：列间 右→左　列内 上→下
```
规则可解决 90% 的情况；AI 仅在同一 Y 坐标范围内出现多列交叉、规则无法判断先后时介入。

## Module 04　Paragraph Recovery
断句、断行、缩进、空行的恢复。当前实现：`merge_broken_sentences`（合并竖排断句，
章节标题不参与合并，避免把标题和正文粘连）。

## Module 05　Dialogue Recovery
```
「」 『』 （） ……
```
恢复为独立段落。当前实现：`restore_dialogue_breaks`（从混合行中拆分叙述+对白）。

## Module 06　Image Reconstruction
```
Page → Illustration → Anchor → Paragraph
```
**Anchor 是关键**：每张图片记录"应插在哪个正文块之后"，而不是全部堆到文末。
当前实现：`Block.image_anchor` 字段 + EPUB Builder 按锚点生成独立插图页。

## Module 07　Chapter Reconstruction
```
Chapter → TOC
```
当前实现：`detect_chapters`（正则匹配序章/第X章/幕間/後記），同步重建 `doc.toc`。

## Module 08　Metadata Reconstruction
来源：封面 OCR、目录 OCR、用户手动输入、AI 建议。最终由用户在 EPUB Builder
页面确认/编辑，不自动覆盖已有手动输入。

## Module 09　Style Reconstruction
```
Vertical (writing-mode: vertical-rl)　Ruby　Indent　LineHeight　CSS
```
当前实现：三套 CSS 模板（denki / mf / web），不做进一步展开。

---

# 第七章　AI Enhancement

AI 参与的场景：
```
OCR 纠错　阅读顺序建议　上下文修复　重复检测　语法修复
```

**硬性约束**：AI 不能直接修改 `Book`。所有 AI 输出必须走 Diff → 用户 Accept/Reject
→ 才写回新版本。这与第十三章的不可变数据流是同一个机制的自然延伸——AI 产生的是
"候选 Book V(n+1)"，而非对 V(n) 的原地编辑。

Prompt 不需要在 SDS 里写全文，采用模板 + Prompt Library（Japanese Novel / Light Novel /
Classical Japanese / Vertical Novel / EPUB Polish / OCR Repair），用户可编辑/导入/导出/分享。

当前状态：**未实现**（Roadmap 第二阶段）。

---

# 第八章　Pipeline

```
Import → OCR → Formatter → AI → EPUB
```
每一步都是可插拔的 Plugin，支持单独关闭/重排序。当前 Formatter 六步已支持
勾选启用/禁用与单步运行，符合这个设计；Pipeline 的保存/加载/分享是 Roadmap
第二阶段的工作。

---

# 第九章　EPUB Builder

重点不是"如何生成合规 EPUB"，而是 **Book 怎么变成 EPUB**：

```
Book → cover.xhtml → images/ → chapter_N.xhtml（按锚点插图）→ nav.xhtml(TOC) → content.opf → .epub
```

竖排（`vertical-rl`）、Ruby、CSS 模板均已实现（见 `builder/epub_builder.py`）。

---

# 第十章　UI（Workspace 模式）

不描述具体控件，只描述工作区职责：

```
Pages      缩略图 / PageType 标注 / 拖动排序 / 右键批量修改
OCR        选择适配器 / 输入(文件夹·单图·PDF) / 识别结果预览
Formatter  Before / After / Diff / Rule / AI建议
AI         Model / Prompt / Temperature / Batch Run / Diff Accept
Images     插图定位 / Anchor 编辑
EPUB       Book Structure 树 / 源码 / 多平台预览(Apple Books/Kindle/Kobo)
Metadata   封面 / 标题 / 作者 / 出版社 / ISBN / 简介 / 标签
Pipeline   Import→OCR→Formatter→AI→EPUB 状态总览
```

当前实现（Tkinter 四标签页：Pages / OCR / Formatter / EPUB）已覆盖前四个工作区的核心
交互（缩略图分类、适配器选择、前后对比、Book Structure 树 + 源码预览）。AI / Images
单独页 / Metadata 独立页 / Pipeline 总览页是差距所在，列入 Roadmap。

审阅认为最终应迁移到 PySide6/Qt 或 SwiftUI 以获得更专业的桌面体验（Dock、Splitter、
Dark Mode、High DPI）。这是合理方向，但属于另起工程量的重写，本轮不做，留在
Roadmap 第三阶段单独立项评估。

---

# 第十一章　Plugin 接口

```
OCRProvider     detect / recognize / supports_vertical / supports_bbox
AIProvider      run(prompt, book) -> Diff
Importer        detect(path) -> bool / to_book(path) -> Book
Exporter        from_book(book) -> bytes
PipelineStep     run(book) -> Book   （已实现，即 Formatter 的六个函数）
PromptTemplate   name / body / editable
EPUBTemplate     css / fonts / margins
```

第五章的 `OCRProvider` 只是这里的一个具体实例；新增任何一类插件都不需要改动
`UnifiedDocument` 或其他工作区的代码。

---

# 第十二章　Roadmap

**第一阶段（已基本完成）**
```
Book（UnifiedDocument）/ Import（文件夹+单图+PDF）/
OCR（Apple Vision）/ Formatter（6步）/ EPUB Builder（3模板）
```

**第二阶段**
```
AI 插件系统 + Prompt Library / 更多 OCR 适配器（pdf-craft/PaddleOCR）/
Pipeline 保存加载 / docx·html·md 导入 / Metadata 独立页 / Book 版本历史 UI（Undo/Redo）
```

**第三阶段**
```
插件市场 / Prompt 社区 / EPUB 模板分享 / 多平台 EPUB 预览 /
UI 迁移评估（PySide6 或 SwiftUI）
```

---

# 第十三章　为什么这样设计

**为什么先页面分类，再 OCR？**
封面、插图、空白页不需要 OCR；对这些页面跑 OCR 反而会把版面噪声当成文字，
破坏本该以图片形式保留的信息。分类在先，OCR 只处理真正含有正文的页面。

**为什么要建立 Book（UnifiedDocument）对象？**
因为 EPUB 生成、AI 校对、Formatter 全部共享同一份数据，如果各模块直接操作字符串，
会出现互相覆盖、无法追溯谁改了什么的问题。Book 对象让每个模块的输入输出都有明确
的契约（schema），新增模块不需要理解其他模块的实现细节。

**为什么每张图片要记录 anchor 而不是位置索引？**
位置索引在前面的处理步骤（合并断句、删除重复段落）之后会失效——段落数量变了，
索引就错位了。Anchor 指向"哪个内容块"，内容块本身在流水线里保持身份不变
（即使文字被合并/修改），所以插图定位可以在任意处理步骤之后依然保持正确。

**为什么 AI 不能直接修改 Book？**
AI 的建议可能是错的，或者只是可选项之一。如果 AI 直接原地覆盖数据，用户就没有
反悔的机会，也无法对比修改前后的差异。走 Diff→Accept/Reject 的流程，本质上是
把"AI 的输出"当成一个新的候选版本，而不是既成事实。

**为什么整条流水线要设计成不可变（Immutable）？**
```
Raw OCR → Book V1 → V2(分类) → V3(阅读顺序) → V4(段落恢复) → V5(AI校对) → V6(EPUB导出)
```
这不是为了"看起来优雅"，而是三个具体的工程收益：
1. **Undo/Redo 免费获得**——每一步都是独立对象，回退只是切换指针。
2. **Diff 天然存在**——V(n) 和 V(n+1) 直接逐块对比即可生成前后对比视图（Formatter
   标签页已经在用这个模式）。
3. **并行/重试安全**——某一步处理失败或效果不理想，重新跑这一步不会污染其他版本，
   也不需要担心"部分修改了一半"的中间状态。

这是本项目**当前实现里最需要补强的一点**：Formatter 各步骤已经在用 `deepcopy` 保证
不改动输入，但版本链没有保留下来、`UnifiedDocument` 也没有 `history` 字段。下面是
具体的补齐实现。

---

# 附：本轮代码改动对照表

| SDS 章节 | 改动文件 | 内容 |
|---|---|---|
| 第三章（Block类型扩展） | `models/document.py` | 新增 9 个页面级 BlockType + `footnote` |
| 第十三章（不可变版本链） | `models/document.py` | `UnifiedDocument` 新增 `history` 字段与 `snapshot()` 方法 |
| 第十三章（Formatter 版本化） | `engine/formatter.py` | `run_pipeline` 每步调用后记录快照，返回值带完整 `history` |
