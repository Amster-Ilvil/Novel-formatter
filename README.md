# Novel Formatter

Novel Formatter 是一款 macOS 桌面排版工具，面向图片/PDF OCR、日文竖排文字处理、多模型 OCR 对比、人工校对、文本格式化和 EPUB 导出。

## 运行环境

- macOS 15 或更高版本
- Python 3.10 或更高版本
- `requirements.txt` 中列出的 Python 包

## 启动程序

推荐使用 macOS 终端运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python gui_pyside6.py
```

也可以双击 `run_novel_formatter.command` 启动。该脚本只负责检查 Python 和 PySide6，然后启动程序，不执行自动更新。

## 主要功能

- 图片和 PDF 输入
- 日文竖排分列 OCR
- Apple Vision、NDLOCR Lite、Manga OCR、PaddleOCR 等 OCR 适配器
- 多模型结果对比和人工裁决
- 图文对照与逐句校对
- Ruby/振假名及残损文字清理
- DOCX、EPUB、TXT 等格式导出
- AI 辅助纠错与排版

OCR 模型只在用户主动选择对应功能时准备，模型权重和缓存不会存放在本仓库中。

## 隐私

本仓库不包含用户文档、OCR 结果、模型权重、运行日志、虚拟环境、本地配置或 API 密钥。

输入文件和导出文件默认保存在本机。只有用户主动选择外部 OCR 或 AI 服务时，相关内容才会发送给对应服务。AI 服务凭据由用户在本机运行时配置，不写入代码仓库。

## 目录说明

- `gui_pyside6.py`：PySide6 图形界面
- `adapters/`：OCR、输入和导出适配器
- `engine/`：对齐、纠错、排版和文档处理逻辑
- `models/`：统一文档数据模型
- `builder/`：DOCX/EPUB 构建器
- `native/`：macOS 原生 OCR 辅助程序源码
- `third_party/`：运行时使用的第三方资源和许可证

## 许可证

使用或再分发第三方资源前，请阅读 `third_party/` 中的许可证和说明文件。

## 参考项目与引用

本项目的接口设计、运行适配或资源来源参考了以下公开项目和官方文档：

- [Qt for Python / PySide6](https://doc.qt.io/qtforpython/)：桌面图形界面。
- [Apple Vision](https://developer.apple.com/documentation/vision)：macOS 原生图像文字识别。
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)：可选的通用 OCR 适配器。
- [NDLOCR-Lite](https://github.com/ndl-lab/ndlocr-lite)：日文 OCR 适配器。
- [Manga Image Translator](https://github.com/zyddnys/manga-image-translator)：48px 日文 OCR 参考实现及模型接口。
- [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)：可选手写识别模型来源。
- [PyMuPDF](https://pymupdf.readthedocs.io/)：PDF 页面和文字层处理。
- [python-docx](https://python-docx.readthedocs.io/)：DOCX 文档处理。
- [Pillow](https://python-pillow.org/)：图像读取、裁切和预处理。
- [jlect-jhr](https://github.com/ZacharyRead/jlect-jhr)：手写输入辅助资源，许可证见 `third_party/jlect_jhr/LICENSE.txt`。

模型权重、第三方服务和外部项目的许可证分别由其原项目负责；本仓库不重新发布模型权重。
