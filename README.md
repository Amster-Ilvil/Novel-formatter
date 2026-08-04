# Novel Formatter

Novel Formatter 是一款 macOS 桌面排版工具，面向图片/PDF OCR、日文竖排文字处理、多模型 OCR 对比、人工校对、文本格式化和 EPUB 导出。

本项目为非盈利、非商业性质，仅用于个人学习、研究和非商业用途。项目不提供商业服务，也不以软件、模型或相关资源进行商业销售。

## 安装教程

### 一、准备系统环境

建议使用 macOS 15 或更高版本，并准备 Python 3.10 或更高版本。Apple Vision 和原生手写识别功能需要 macOS 的系统框架；如果要编译原生辅助程序，还需要 Xcode Command Line Tools。

检查系统版本和 Python：

```bash
sw_vers
python3 --version
```

如果没有 Python，可从 [Python 官方网站](https://www.python.org/downloads/macos/) 安装。安装完成后重新打开终端，再执行上面的版本检查。

### 二、下载源码

在 GitHub 页面点击 **Code → Download ZIP** 解压，或使用 Git：

```bash
git clone https://github.com/Amster-Ilvil/Novel-formatter.git
cd Novel-formatter
```

不要把模型权重、虚拟环境或生成的 EPUB 文件复制到源码目录后再提交或重新打包。

### 三、创建独立 Python 环境

在项目目录打开终端，创建项目专用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

看到命令行前出现 `.venv` 后，说明虚拟环境已经启用。以后每次打开新终端运行程序前，都需要先执行：

```bash
cd /项目所在目录/Novel-formatter
source .venv/bin/activate
```

退出虚拟环境：

```bash
deactivate
```

如果只需要打开界面，也可以只安装核心依赖；但完整的 `requirements.txt` 能启用 PDF、DOCX、EPUB、AI 和图像处理功能。

### 四、启动程序

#### 方式 A：使用终端启动

```bash
source .venv/bin/activate
python gui_pyside6.py
```

#### 方式 B：双击启动脚本

双击项目目录中的 `run_novel_formatter.command`。如果 macOS 阻止执行，打开终端运行：

```bash
chmod +x run_novel_formatter.command
./run_novel_formatter.command
```

该脚本只检查 Python、虚拟环境和 PySide6，然后启动程序，不执行项目自动更新。

#### 方式 C：运行命令行排版入口

处理图片目录并导出 EPUB：

```bash
source .venv/bin/activate
python run.py /path/to/images/ output.epub --title "书名" --author "作者"
```

输入路径和输出路径可以替换成实际文件夹或文件名。带空格的路径必须使用引号包裹。

### 五、配置 Apple Vision 原生助手

普通 Apple Vision OCR 可直接使用系统能力。若要启用项目附带的原生识别辅助程序，先安装 Xcode Command Line Tools：

```bash
xcode-select --install
```

然后在项目目录执行：

```bash
chmod +x build_apple_vision_helper.command
./build_apple_vision_helper.command
```

编译成功后重新启动程序。若当前 macOS 或 Xcode 不支持对应原生接口，程序会使用可用的回退路径，不影响其他 OCR 功能。

### 六、准备可选 OCR 模型

模型权重不包含在 GitHub 仓库中。第一次使用某个 OCR 引擎时，程序会在本机准备对应运行环境或模型缓存。

可选模型包括：

- Apple Vision：使用 macOS 系统能力，通常不需要额外模型下载。
- NDLOCR Lite：首次使用时准备对应 OCR 运行文件。
- Manga OCR 48px：首次使用时下载模型权重，所需空间和时间取决于网络情况。
- PaddleOCR：首次选择 PaddleOCR 时创建独立运行环境并准备模型。
- YomiToku：按界面提示准备其运行环境和权重。

48px 模型也可以提前执行：

```bash
chmod +x prepare_48px_ar.command
./prepare_48px_ar.command
```

模型准备过程中不要删除项目的本地缓存目录，也不要同时启动多个 OCR 任务。下载失败时重新打开程序，通常会从已完成的部分继续校验或重新准备。

### 七、第一次使用流程

1. 启动 `gui_pyside6.py` 或 `run_novel_formatter.command`。
2. 在页面管理中导入图片文件夹或 PDF。
3. 根据书籍版面选择普通 OCR、分列 OCR 或多模型 OCR。
4. 首次使用某个模型时，等待模型环境准备完成。
5. OCR 完成后，在 OCR 对比工作区检查模型差异。
6. 在图文对照工作区核对低置信度文字、Ruby 注音和残损字符。
7. 应用融合结果后，再进入 Formatter 和 EPUB 导出。

建议先用少量页面测试输入、分列和导出流程，确认设置正确后再处理整本书。

### 八、常见问题

**提示找不到 PySide6**

确认虚拟环境已启用，并使用同一个 Python 安装依赖：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import PySide6; print(PySide6.__version__)"
```

**双击脚本后窗口立即关闭**

从终端启动以查看错误：

```bash
./run_novel_formatter.command
```

如果脚本没有执行权限，先运行 `chmod +x run_novel_formatter.command`。

**模型下载卡住或失败**

先退出当前 OCR，检查网络和磁盘空间，再重新启动。不要把下载中的临时文件当作完整模型使用。Apple Vision 可以作为临时替代引擎验证输入和分列是否正确。

**OCR 结果为空或分列不正确**

先用少量页面检查正文区域、文字方向和分列设置。日文竖排书籍应确认页面方向和列顺序正确；带 Ruby 或邻列残影的页面应启用 Ruby 过滤和正文区域掩膜。

**导出 EPUB 失败**

确认输出目录可写、磁盘空间充足，并关闭正在打开同名 EPUB 的其他程序。先导出少量页面验证，再处理完整书籍。

### 九、卸载项目环境

删除项目目录即可移除源码和项目虚拟环境：

```bash
rm -rf .venv
```

OCR 模型缓存由各模型适配器保存到本机缓存位置。删除缓存前请确认不再需要已下载的模型；删除缓存不会影响源码。

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

本项目的发布和使用均以非盈利、非商业为前提；第三方项目和资源仍适用其各自的许可证与使用条款。

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
