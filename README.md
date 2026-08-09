# Novel Formatter

面向日文竖排书籍的 OCR、校对、排版与 EPUB 制作工具，支持 macOS 和 Windows。

## 功能

- 图片、文件夹、PDF 导入
- 日文竖排 OCR 与分列识别
- 多模型 OCR 对比与融合
- 图文对照、低置信度复核与人工校对
- Ruby、页眉页码、跨列/跨页文本处理
- Formatter 文本整理
- EPUB 导出
- AI 修复包导出，可交给大模型继续纠错与排版

## OCR 引擎

支持或可选：

- Apple Vision（macOS）
- NDLOCR
- Manga OCR
- PaddleOCR
- YomiToku

Apple Vision、Swift OCR Helper、Apple Pencil 手写识别仅在 macOS 可用。

## 安装

推荐使用 Python 3.10 或更高版本。

### macOS

```bash
git clone https://github.com/Amster-Ilvil/Novel-formatter.git
cd Novel-formatter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python gui_pyside6.py
```

也可以直接运行：

```bash
./run_novel_formatter.command
```

### Windows

```powershell
git clone https://github.com/Amster-Ilvil/Novel-formatter.git
cd Novel-formatter
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python gui_pyside6.py
```

安装包可直接从 [Releases](https://github.com/Amster-Ilvil/Novel-formatter/releases) 下载。

## 推荐流程

1. 导入 PDF、图片或图片文件夹。
2. 标记封面、扉页、目录、插图、正文、后记等页面类型。
3. 在 OCR 页面选择模型，并根据版面开启分列。
4. 运行 OCR。
5. 在 OCR 对比和图文对照中处理模型分歧、漏字、Ruby、页码和跨页连接。
6. 应用融合结果并进行 Formatter 整理。
7. 直接导出 EPUB，或导出 AI 修复包继续处理。

## AI 修复包

AI 修复包用于把 OCR 证据、文本、结构和资源交给 GPT、Claude 等大模型继续复核。

可以用于：

- OCR 错字、漏字和标点修复
- 人名、地名、术语一致性检查
- 跨页、跨章节语境复核
- EPUB 结构与排版检查
- 生成接近出版成品的最终 EPUB

## 模型与本地数据

OCR 模型权重、缓存、虚拟环境、日志、数据库和用户输出不会作为项目源码提交。

首次使用部分 OCR 引擎时，程序会在本机准备所需依赖或模型文件。

仓库启用了隐私审计，PR 和 `main` 更新会自动检查常见密钥、本机用户路径、运行缓存和其他不应提交的数据。

## 平台说明

| 功能 | macOS | Windows |
|---|---:|---:|
| 主界面 | ✓ | ✓ |
| PDF / 图片 OCR | ✓ | ✓ |
| 多模型 OCR | ✓ | ✓ |
| Formatter / EPUB | ✓ | ✓ |
| Apple Vision | ✓ | — |
| Swift OCR Helper | ✓ | — |
| Apple Pencil 手写识别 | ✓ | — |

## 用途

本项目为非盈利、非商业性质，仅用于个人学习、研究和非商业用途。

第三方 OCR、模型和依赖仍适用其各自许可证与使用条款。
