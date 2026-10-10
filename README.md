<div align="center">

**简体中文** · [English](README_en.md) · [日本語](README_ja.md)

</div>

<p align="center">
  <img src="assets/novel_formatter_banner.webp" alt="Novel Formatter" width="100%">
</p>

<h1 align="center">Novel Formatter</h1>

<p align="center"></p>

<p align="center">
  <img alt="macOS" src="https://img.shields.io/badge/macOS-Apple%20Silicon%20%2F%20Intel-black?logo=apple">
  <img alt="Windows" src="https://img.shields.io/badge/Windows-x64-0078D4?logo=windows11&logoColor=white">
  <img alt="PySide6" src="https://img.shields.io/badge/UI-PySide6-41CD52?logo=qt&logoColor=white">
  <img alt="Japanese OCR" src="https://img.shields.io/badge/OCR-Japanese-2f6feb">
  <img alt="EPUB" src="https://img.shields.io/badge/Output-EPUB-8A2BE2">
</p>

> 日文竖排 OCR · 多模型对比与融合 · 图文对照 · Formatter · EPUB · AI 修复包 · Apple Vision



## 主要特性

- **图片 / PDF 导入**：支持图片、文件夹和 PDF 页面处理。
- **日文竖排 OCR**：针对日文书籍纵排、分列和跨列内容处理。
- **多模型 OCR 对比与融合**：可组合 Apple Vision、Windows Snipping OCR、NDLOCR-Lite、Hayai OCR、48px OCR、PaddleOCR 等结果进行复核。
- **图文对照校对**：结合原始页面与 OCR 结果检查错字、漏字、低置信度文本和版面问题。
- **Ruby / 页眉页码处理**：使用 FindText CenterNet 辅助定位并清理假名注音、页眉、页码、跨列和跨页残片。
- **Formatter 文本整理**：对 OCR 文本进行段落、标点、标题和跨页接续等后处理。
- **EPUB 制作**：从整理后的正文、结构和资源直接导出 EPUB。
- **AI 修复包**：导出 OCR 证据、正文、结构和资源，交给 GPT、Gemini、GLM、DeepSeek 等兼容服务继续复核并生成接近出版成品的 EPUB。
- **本地数据优先**：模型、缓存、日志、数据库和用户输出不作为项目源码提交。


## v2.0 主要更新

相比 v1.3，v2.0 对 OCR、校对、PDF 与 EPUB 工作流进行了较大升级：

- **全新的项目工作区**：统一保存页面、OCR、裁决、格式处理与 EPUB 制作状态，支持项目恢复与最近项目管理。
- **多模型 OCR 与裁决升级**：完善逐列 / 全页等模型角色、分歧复核、图文对照和人工裁决流程，并优化 Hayai OCR、NDLOCR-Lite、48px OCR 等本地引擎的运行链路。
- **原生系统 OCR 扩展**：新增 Windows 11 Snipping OCR；Apple OCR 重构为更稳定的原生 Vision 路径，并改善 Intel 与 Apple Silicon Mac 的兼容性。
- **PDF 文字层流程重构**：直接使用“页面管理”中的原始 PDF 与物理页映射，改善竖排阅读顺序、跨页接续、段落结构和出版级文本整理。
- **AI 裁决增强**：支持 GPT、Gemini、GLM、DeepSeek 等兼容服务，并加强裁决包、模型证据与人工修改的安全往返。
- **EPUB / TEI 与界面升级**：增强封面、插图、分页、元数据和 EPUB 结构处理，同时重构主要界面、预览与进度反馈，并提升长篇任务稳定性和跨平台运行能力。

## OCR 引擎

支持或可选：

- Apple Vision（macOS）
- Windows Snipping OCR（Windows 11）
- NDLOCR-Lite
- Hayai OCR
- 48px OCR
- PaddleOCR
- Paddle VL MLX（macOS Apple Silicon）
- FindText CenterNet（Ruby 区域辅助）

Apple Vision、Swift OCR Helper、Apple Pencil 手写识别仅在 macOS 可用；Windows Snipping OCR 仅在支持相应系统 OCR 组件的 Windows 11 环境可用。

## 推荐流程

1. 导入 PDF、图片或图片文件夹。
2. 标记封面、扉页、目录、插图、正文、后记等页面类型。
3. 在 OCR 页面选择模型，并根据版面开启分列。
4. 运行 OCR。
5. 在 OCR 对比和图文对照中处理模型分歧、漏字、Ruby、页码和跨页连接。
6. 应用融合结果并进行 Formatter 整理。
7. 直接导出 EPUB，或导出 AI 修复包继续处理。

## AI 修复包

AI 修复包用于把 OCR 证据、文本、结构和资源交给大模型继续复核，而不是只提供一份脱离版面的纯文本。

可以用于：

- OCR 错字、漏字和标点修复
- 人名、地名、术语一致性检查
- 跨页、跨章节语境复核
- EPUB 结构与排版检查
- 生成接近出版成品的最终 EPUB

## 安装与启动

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

## 平台说明

| 功能 | macOS | Windows |
|---|---:|---:|
| 主界面 | ✓ | ✓ |
| PDF / 图片 OCR | ✓ | ✓ |
| 多模型 OCR | ✓ | ✓ |
| Formatter / EPUB | ✓ | ✓ |
| AI 修复包 | ✓ | ✓ |
| Apple Vision | ✓ | — |
| Windows Snipping OCR | — | Windows 11 |
| Swift OCR Helper | ✓ | — |
| Apple Pencil 手写识别 | ✓ | — |

## 模型与本地数据

OCR 模型权重、缓存、虚拟环境、日志、数据库和用户输出不会作为项目源码提交。

首次使用部分 OCR 引擎时，程序会在本机准备所需依赖或模型文件。


## 参考项目与引用

本项目的界面、OCR 适配、文档处理和手写识别能力参考或使用了以下公开项目与官方文档：

- [Qt for Python / PySide6](https://doc.qt.io/qtforpython/)：桌面 GUI 框架。
- [Apple Vision](https://developer.apple.com/documentation/vision)：macOS 原生 OCR 与视觉识别能力。
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)：通用 OCR 引擎与跨平台识别参考。
- [NDLOCR-Lite](https://github.com/ndl-lab/ndlocr-lite)：日文书籍 OCR 与版面识别参考。
- [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)：可选手写识别模型与推理参考。
- [PyMuPDF](https://pymupdf.readthedocs.io/)：PDF 页面、图像与文字层处理。
- [python-docx](https://python-docx.readthedocs.io/)：DOCX 读取与生成。
- [Pillow](https://python-pillow.org/)：图像读取、裁切与预处理。
- [jlect-jhr](https://github.com/ZacharyRead/jlect-jhr)：日文手写识别辅助资源；许可证见 `third_party/jlect_jhr/LICENSE.txt`。

第三方项目、模型和资源仍适用其各自许可证；本仓库不重新发布 OCR 模型权重。

## Star History

<p align="center">
  <a href="https://www.star-history.com/?type=date&repos=Amster-Ilvil%2FNovel-formatter">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date&theme=dark">
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date">
      <img alt="Novel Formatter Star History Chart" src="https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date" width="800">
    </picture>
  </a>
</p>

## 用途与许可

本项目原创代码与文档以 [MIT License](LICENSE) 开源，可在遵守许可证的前提下用于个人、学习、研究或商业用途。

第三方 OCR、模型、依赖和平台能力仍适用其各自许可证与使用条款；完整致谢见 [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)。
