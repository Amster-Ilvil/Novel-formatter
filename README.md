# Novel Formatter

Novel Formatter 是一款适配 macOS 和 Windows 的桌面排版工具，面向图片/PDF OCR、日文竖排文字处理、多模型 OCR 对比、人工校对、文本格式化和 EPUB 导出。

Windows 已支持源码安装和主程序运行。macOS 专用的 Apple Vision、Swift OCR Helper 和 Apple Pencil 手写识别功能仅在 macOS 上可用；Windows 用户可以使用 NDLOCR、Manga OCR、PaddleOCR、YomiToku 等跨平台 OCR 引擎。

本项目为非盈利、非商业性质，仅用于个人学习、研究和非商业用途。项目不提供商业服务，也不以软件、模型或相关资源进行商业销售。

## 安装教程

### 一、准备系统环境

macOS 建议使用 macOS 15 或更高版本，并准备 Python 3.10 或更高版本。Windows 建议使用 Windows 10/11 64 位系统，并准备 Python 3.10 或更高版本。

Apple Vision 和原生手写识别功能需要 macOS 的系统框架；如果要编译原生辅助程序，还需要 Xcode Command Line Tools。Windows 不需要安装 Xcode。

macOS 检查系统版本和 Python：

```bash
sw_vers
python3 --version
```

如果没有 Python，可从 [Python 官方网站](https://www.python.org/downloads/macos/) 安装。安装完成后重新打开终端，再执行上面的版本检查。

Windows 用户可从 [Python 官方网站](https://www.python.org/downloads/windows/) 安装 Python。安装时务必勾选 **Add Python to PATH**，然后在 PowerShell 中检查：

```powershell
py --version
python --version
```

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

### 四、Windows 安装与启动

Windows 用户在项目目录打开 PowerShell，执行：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python gui_pyside6.py
```

如果 PowerShell 阻止虚拟环境脚本运行，可仅对当前用户允许本地脚本：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后重新执行：

```powershell
.\.venv\Scripts\Activate.ps1
python gui_pyside6.py
```

也可以不使用 PowerShell 激活虚拟环境，直接调用虚拟环境中的 Python：

```powershell
.\.venv\Scripts\python.exe gui_pyside6.py
```

Windows 不使用项目中的 `.command` 启动脚本；这些脚本用于 macOS。首次使用 OCR 模型时，程序会在本机创建或准备对应的模型运行环境，模型权重不会上传到 GitHub。

### 五、启动程序

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

### 六、配置 Apple Vision 原生助手

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

### 七、准备可选 OCR 模型

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

### 八、第一次使用流程

1. 启动 `gui_pyside6.py` 或 `run_novel_formatter.command`。
2. 在页面管理中导入图片文件夹或 PDF。
3. 根据书籍版面选择普通 OCR、分列 OCR 或多模型 OCR。
4. 首次使用某个模型时，等待模型环境准备完成。
5. OCR 完成后，在 OCR 对比工作区检查模型差异。
6. 在图文对照工作区核对低置信度文字、Ruby 注音和残损字符。
7. 应用融合结果后，再进入 Formatter 和 EPUB 导出。

建议先用少量页面测试输入、分列和导出流程，确认设置正确后再处理整本书。

## 推荐工作流程：从原始 PDF 到 EPUB

本项目目前主要面向 **日文竖排文库小说**，。推荐按照下面的流程操作。整本约 400 页的书籍，OCR 通常需要 30–60 分钟；从导入到 EPUB 成品，整体约需 1–2 小时，具体时间取决于 OCR 模型、设备性能、页面复杂度和人工复核范围。

### 1. 加载图片或文件夹

在“页面管理”中加载 PDF、图片文件或图片文件夹。导入后先确认页数、页面顺序和图像方向是否正确。

### 2. 标记页面类型

为相应页面标记以下类型：

- 封面
- 扉页
- 目录
- 插图
- 空白页
- 后记
- 版权页

只有标记为正文的页面会进入 OCR。这样可以避免把封面文字、目录、页码或插图中的文字误当作正文，也能让最终 EPUB 更准确地保留原书结构和资源。

### 3. 设置多模型 OCR 和识别区域

进入“OCR 识别”界面：

1. 勾选需要使用的多个 OCR 模型，推荐使用 **NDLOCR 作为底稿模型**；
2. 开启“分列”，让程序按照日文竖排版面逐列识别；
3. 在右侧页面预览中用鼠标手动框选正文区域；
4. 将页眉、页码、装饰文字和其他不属于正文的内容排除在框选区域之外。

正文区域框选得越准确，后续的多模型对比、句子融合和 AI 修复越稳定。遇到带 Ruby 注音、邻列残影或边缘残损文字的页面，应优先缩小正文区域或启用相应的 Ruby 过滤与图像清理选项。

### 4. 运行 OCR

开始 OCR 后，程序会按照页面类型、物理列和所选模型执行识别。一本约 400 页的书通常需要 30–60 分钟，实际时间会受以下因素影响：

- 使用的 OCR 模型数量；
- 是否启用分列、整句复核和逐字审校；
- Mac 的 CPU、GPU、内存和磁盘速度；
- 页面分辨率、文字密度和插图数量。

长篇任务建议保持设备接通电源，并避免同时运行多个 OCR 任务。若需要停止，应等待当前页面或当前模型任务响应取消状态。

### 5. 进行 OCR 对比和图文对照

OCR 完成后，进入“OCR 对比”和“图文对照”界面进行人工裁决。程序会展示不同模型的识别结果、融合候选、原始页面和当前正文，便于检查：

- 模型之间的分歧；
- 漏字、错字和错误标点；
- Ruby 注音、邻列残片和残损字符；
- 页眉、页码等非正文内容是否混入；
- 跨列、跨页和对白的连接是否正确。

一本书可能需要处理几百到几千处候选差异。可以优先审核低置信度、模型分歧较大以及包含特殊排版的句子。

### 6. 导出 AI 修复包并生成 EPUB

建议在 OCR 对比完成后导出 **AI 修复包**，交给 GPT、Claude 等大模型进行出版级修复。推荐两种使用方式：

**方式一：直接生成 EPUB**

将 AI 修复包交给大模型，并要求模型依据包内的 OCR 证据、页面结构、图片资源和排版信息，直接生成出版级 EPUB。

**方式二：两阶段纠错后回导程序**

1. 将项目 GitHub 地址和 AI 修复包一并提供给大模型；
2. 要求大模型先根据项目格式生成可导入的 AI 纠错 JSON；
3. 在程序中导入 AI 纠错结果；
4. 在 OCR 对比和图文对照界面人工审核导入结果；
5. 导出融合结果和 EPUB 骨架；
6. 再将融合结果、骨架和相关说明交给大模型生成最终 EPUB。

第二种方式相当于让大模型分别完成“文字纠错”和“出版排版”两次处理，通常更容易保持文字、章节、插图、封面和竖排布局的一致性。根据实际书籍和模型表现，二次处理可能带来约 0.3%–0.5% 的结果改善，但最终效果仍取决于输入质量、模型能力和人工审核。

在正文区域、页面分类和 OCR 模型设置正确的前提下，这套流程通常可以将整本书的文字正确率提升到约 99% 的水平，同时较完整地保留封面、插图、目录和原书排版信息。这里的数值是实际使用中的经验值，不代表对所有 PDF、模型或设备的固定保证。

## 常见问题

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

## 卸载项目环境

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
