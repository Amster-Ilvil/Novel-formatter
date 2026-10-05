# Novel Formatter

Novel Formatter 是面向扫描书、日文小说 OCR、文本校订和 EPUB 制作的桌面工具。此仓库版本已按公开发布要求清理：不包含开发电脑用户名、绝对路径、邮箱、访问令牌、运行日志、调试输入、模型缓存或虚拟环境。

当前已适配 macOS 与 Windows 安装，Windows 可直接双击 `启动Windows.bat` 完成首次部署并启动程序。

## 首次部署

### Windows

双击 `启动Windows.bat`。

启动器会依次完成：

1. 检测 Windows 架构和可用 Python；没有兼容 Python 时，自动下载并校验便携 Python。
2. 创建项目独立环境，不修改系统 Python。
3. 从国内镜像优先安装**主程序界面与文档处理依赖**。
4. **不安装任何 OCR 专用依赖，不下载任何 OCR 模型。**
5. 完成后直接启动程序。

Windows 下载顺序为 BITS、`curl.exe`、PowerShell HTTPS；下载失败会切换来源。

### macOS

双击 `启动Mac.command`。首次运行会构建应用外壳，只准备独立 Python 和主程序界面依赖，不下载 OCR 专用依赖或模型。Apple Silicon 与 Intel Mac 分别下载对应架构的运行环境。

macOS 不再扫描桌面更新包，也不会自动替换应用代码。

### 命令行

已有 Python 3.10–3.13 时可运行：

```bash
python bootstrap.py --install-main-deps --launch
```

旧版的 `--prepare-models`、`--profile` 和 `NOVEL_FORMATTER_BOOTSTRAP_PROFILE` 仅为命令兼容保留；即使传入，也不会在启动阶段下载 OCR 资源。

## 下载源

默认按可达性选择并自动回退：

- Python 包：清华大学 TUNA → 阿里云 → 华为云 → PyPI 官方。
- Hugging Face 模型：HF-Mirror → Hugging Face 官方。
- Paddle 模型：ModelScope 优先，再尝试 BOS、AIStudio 和 Hugging Face。
- 独立 Python：南京大学 GitHub Release 镜像 → GitHub 官方。
- NDLOCR-Lite：上游没有确认可靠的国内官方镜像，保留官方 GitHub 来源。

所有有固定哈希的运行环境或模型均在替换前校验。更多说明见 `docs/DOWNLOAD_SOURCES.md`。

## OCR 按需安装与手动更新边界

首次部署不会安装任何 OCR 专用依赖，也不会下载任何 OCR 模型。用户点击“开始 OCR”后，程序先检查当前所选模型；发现缺失时显示所需组件和用途，只有用户点击“安装并继续”才会创建独立环境并下载对应资源。点击取消不会下载。

已经完整安装的 OCR 模型不会在启动时联网检查，不会被静默替换，也不会自动升级。后续模型版本检查和更新仍只能在：

`系统设置 → OCR 设置 → 管理 OCR 模型更新`

中由用户手动触发并确认。


## PaddleOCR · AI Studio API（云端）

在现有本地 PaddleOCR 之外，新增独立的 **PaddleOCR · AI Studio API** OCR 引擎。它不会替换本地模型，可作为主模型或多 OCR 的模型 2/3。

- 同时支持 AI Studio PaddleOCR 任务页复制的**模型专属同步 API URL**，以及官方 `POST /api/v2/ocr/jobs` **异步 v2**。
- 同步模式使用 `Authorization: token ...` + Base64 图片；异步模式使用 Bearer Token + job 轮询。
- 云端引擎固定按**整页**上传，不走日文物理分栏的“逐列远程调用”，避免页面额度成倍消耗并保留云端版面上下文。
- PP-OCR 的 `rec_texts/rec_scores/rec_polys/rec_boxes`、Structure/VL 的 `parsing_res_list` 和 markdown fallback 都会转换成统一 OCR block。
- Token 不进入项目/模式 JSON；macOS 可保存到 Keychain，Windows 使用 DPAPI，Linux 不退回明文文件。优先支持官方环境变量 `PADDLEOCR_ACCESS_TOKEN`，并兼容旧的 `AISTUDIO_ACCESS_TOKEN`。
- 携带 Token 的 URL 被限制为 HTTPS `*.aistudio-app.com`，预签名结果 URL 下载时不发送 Token。
- 异步模型跟随官方 2026 SDK：PP-OCRv6 / PP-OCRv5 / PP-OCRv5-latin / PP-StructureV3 / PaddleOCR-VL / VL-1.5 / VL-1.6；新配置默认 PP-OCRv6。
- 429 会区分“当日额度耗尽”和“请求频率过高”；HTTP 503/504 与 v2 `10010/12002` 走可取消的指数退避。异步 multipart 重试使用可重放字节，避免第二次上传空文件。

完整配置、接口模式和安全说明见 `docs/PADDLE_AISTUDIO_API.md`。云端提交采用配额安全重试：连接前失败可重试，但请求已发出后的超时/5xx 不盲目重提；异步 JSONL 采用 64 MB 流式上限，所有远端页失败时该云模型会自动从多模型共识中隔离。

## Hayai OCR v2.5-nova 接入

日文竖排 OCR 链已新增 **Hayai OCR v2.5-nova**。它在本项目中被严格定位为“文字 crop 识别器”，不负责整页文字检测或版面分析：

- GUI 可直接选择 `Hayai OCR v2.5-nova`，也可作为多模型 OCR 的模型 2/3。
- 选择后强制启用固定正文区与物理分列，整页图片不会直接交给 Hayai。
- 复用现有右→左列顺序、Ruby/碎片清理、共享 crop、句子重识别、多模型融合和人工复核。
- 使用独立 `.venv-hayai-ocr` 与 `.model-cache/hayai-ocr`，不污染 Paddle/48px 等其它 OCR 环境。
- 默认固定 `hayai-ocr==2.3.0`；支持 PyTorch 的 CUDA/MPS/CPU、INT8/INT4，以及可选 LiteRT 后端；Torch/LiteRT 分别确认运行时与权重完整性。
- 运行时 readiness 同时校验固定包版本、后端类型与对应权重；旧版本环境、过期状态标记或“只装了另一后端”的缓存都不会被误判为可用。
- 请求不存在的 CUDA/MPS 时安全回退 CPU；INT4/INT8 不受当前 PyTorch/torchao 支持时自动回退非量化，优先保证 OCR 可用性。
- INT4/INT8 不可用时自动降级为非量化 Torch，不让性能选项变成功能故障。
- 对生成式 OCR 增加长度、缺字和 CJK 字符比例检查；其置信度在融合层只按启发式证据处理，不冒充模型概率。
- 使用较长的列内 crop，并通过常驻 worker 真正批量识别；Hayai 专用估字分段开关不会改变 48px 的切块行为。
- 首次成功识别后普通启动离线复用 Hayai 专用缓存，不在后台静默检查/替换上游模型快照。

首次选择 Hayai OCR 时仍遵循本项目的按需安装规则：只有点击“安装并继续”后才创建独立环境并下载模型。

## Apple M6 性能档

新版会自动识别 Apple M6，并对 OCR 整条流水线启用 M6 专用运行时策略：Hayai/48px 的 PyTorch 环境优先使用 MPS，PaddleOCR/NDLOCR 保持已验证后端并提升 CPU 预处理，PaddleOCR-VL 优先走 MLX-VLM；页面预处理并发提高到 8、图像编码提高到 4，重型 MPS/MLX 模型保持单实例以避免统一内存争抢。详细参数见 `docs/M6_APPLE_SILICON.md`。

## 隐私与 Git 发布

- `.gitignore` 排除虚拟环境、模型、运行状态、日志、调试输入、用户文档和密钥文件。
- API 密钥只应在程序运行时填写，不能写入源码或提交 `.env`。
- `tools/public_release_audit.py` 会扫描发布树中的本机路径、个人邮箱、常见令牌、私钥和运行时文件。
- 发布前运行：

```bash
python tools/public_release_audit.py --root . --write-report PUBLIC_RELEASE_AUDIT.json
```

详细规则见 `docs/PRIVACY.md`。

## OCR 代码保护

启动准备、按需安装确认和下载源选择位于独立基础设施与运行环境检测层中，不修改 OCR 识别算法。`OCR_CODE_FREEZE_20260804.json` 保存受保护 OCR 文件与 GUI OCR 区段的哈希，可用于发布前复核。

## 开发与测试

```bash
python -m pytest -q
```

公开仓库不包含模型权重和 OCR 运行环境。首次部署仅在缺少 Python 或主程序依赖时需要联网；OCR 资源仅在开始识别并确认后按需下载。

## 第三方组件

项目依赖多个第三方 OCR、模型和 Python 包。公开发布或再分发前，应分别核对相应上游许可证、模型许可和使用条款；本仓库不会用统一声明覆盖第三方许可。

## 可选：保留日文 Ruby（findtextCenterNet）

日文竖排 OCR 页面新增 **“保留原文 Ruby（findtextCenterNet）”** 开关，默认关闭。

- **关闭（默认）**：不检查 findtextCenterNet 运行时、不下载源码/模型、不增加推理时间；现有 Apple Vision、48px、NDLOCR、PaddleOCR、Hayai 等 OCR 路径保持原样。
- **开启**：普通 OCR 仍只识别 Ruby-free 的分列正文；侧边 Ruby 在普通 OCR 临时列图中被排除。分列阶段同时免费记录“疑似 Ruby”的**几何坐标（不含任何识别文字）**。默认的“智能 ROI”会把相邻正文列与这些小字候选合成少量原图上下文框，`findtextCenterNet` **只 OCR 这些 ROI**，不再默认整页重扫；高级选项仍可切换“全页精确扫描（慢）”。
- 两条通道严格隔离：findtextCenterNet **不参与 OCR 对比、票数、字符融合，也不修改任何源 OCR 文档**；Ruby 只作为 metadata overlay 写入最终 fused/权威文档。`block.text`、`block.ocr_raw`、`block.type` 都继续保持普通正文证据。重复底字会保存只读上下文锚点；AI 回写后只有上下文能唯一证明具体 occurrence 时才恢复，否则跳过，不猜。
- Smart ROI 会按 findtext 的 768×768 实际 tile 成本安全收边，避免“只超几像素却多跑一个 detector window”；多 OCR Ruby 候选按**原页物理列**而不是数组下标合并。成功 ROI 结果有独立缓存，重复 Fusion/AI roundtrip 可直接复用；单个 ROI 失败时自动隔离重试，不影响其他 Ruby，更不会影响主 OCR。
- EPUB 使用原生 EPUB3 `<ruby><rt>…</rt></ruby>`，CSS 使用 `ruby-position: over`：横排显示在汉字上方，日文竖排按书写方向显示在正文外侧。
- 首次开启才会创建 `.venv-findtext-centernet`，并在 `.ocr-runtimes/findtext-centernet` 准备固定 commit 上游源码。**后端严格跟随上游 `run_ocr.py` 的 CoreML → ONNX → Torch 约定**：macOS 首次安装默认准备官方 CoreML 三件套（约 618 MiB），Windows/Linux 默认准备量化 ONNX 三件套（约 655 MiB）；如果目录中已经有任一完整上游后端则直接复用，不强制再下载另一套。Torch 的 `model.pt` + `model3.pt`（约 1.42 GiB）仅作为现有安装/显式选择/回退路径。
- Hugging Face 官方模型仓库已使用 Xet；Xet 现在只作为快速通道：程序直接监控 Hugging Face `*.incomplete` 缓存是否真实增长，默认连续约 45 秒无文件活动即终止 Xet 并立即回退 `.part` 可续传 Python HTTPS / 系统 `curl`。`416` Range 探测会按非致命状态处理；一次 Xet 停滞后进入临时冷却，后续 CoreML/ONNX 包直接走 HTTP，避免每个文件都重复卡住。旧版 `.part` 仍保留并续传。
- 网络受限时仍可用 `NOVEL_FORMATTER_FINDTEXT_BACKEND=torch|onnx|coreml` 强制后端；旧版 Torch 本地文件与 `.part` 兼容逻辑继续保留。macOS/Linux 自动用 `make` 编译 `linedetect`；Windows 按上游 `Makefile.mak` 需要 Microsoft C++ Build Tools（`nmake`/`cl`）。
- 命令行可使用 `--preserve-ruby --ruby-scan-mode smart_roi`（默认）；需要最大召回诊断时可改为 `--ruby-scan-mode full_page`。不指定 `--preserve-ruby` 时行为与旧版一致。

实现固定到已验证的 findtextCenterNet commit `295bc88703039f9a83eef8146c743c94cf728b96`，避免上游后续改动静默改变 Ruby 行为。


## OCR Runtime v2.3：Hayai 缓存复用与 findtext 统一运行时

- Hayai OCR 现在会复用项目缓存、`HF_HOME`、`HUGGINGFACE_HUB_CACHE` 与标准 `~/.cache/huggingface` 中已经完整下载的模型，并在启动 OCR 前自动刷新旧的“未就绪”探测结果；模型完整时不再重复弹“安装并继续”。
- Hayai readiness 检测与实际 worker 使用同一个缓存根；并按 Hayai v2.1 上游真实加载逻辑判断：SigLIP2 仓库只要求图像 processor 资源，文本 tokenizer 从 Hayai 自己的模型仓库判断，不再因为 SigLIP2 目录里没有 tokenizer 而反复弹“安装并继续”。
- findtextCenterNet Ruby 专家改用公共 `ensure_venv()` 与 Runtime Catalog 生命周期；固定 commit 源码优先 git 获取并精确校验，git 不可用时才回退可续传 ZIP。
- findtextCenterNet 仍只处理 Smart ROI / Ruby 结构，不参与正文 OCR compare、Fusion 或 AI 正文裁决。

详见 `OCR_RUNTIME_UNIFICATION_HAYAI_PROMPT_FIX_20260904.md`。

## findtextCenterNet Ruby runtime v2.4

- Large `model.pt` / `model3.pt` downloads now emit periodic progress (MiB, %, speed, ETA), so normal transfers no longer look like a stalled OCR job.
- Standard Hugging Face caches are probed and reused before any network download.
- Python HTTPS preserves/resumes `.part` files and uses a shorter no-data timeout; system curl fallback also emits keepalive progress.
- See `FINDTEXT_MODEL_DOWNLOAD_PROGRESS_FIX_20260904.md`.


## findtextCenterNet upstream-native adapter v2.5

- Integration now follows the upstream project boundary: Novel Formatter does not reimplement Detector, Transformer, Ruby classification, or JSON generation.
- A persistent bridge worker imports the pinned upstream `run_ocr.py` unchanged and reuses its single `OCR_Processer` for all Smart ROIs in the book.
- The worker returns upstream JSON verbatim; Novel Formatter only handles ROI planning, original-page coordinate restoration, locked Ruby metadata, and EPUB output.
- If the persistent worker fails, remaining ROIs fall back to the upstream documented `run_ocr.py <image>` CLI; the main OCR text remains fail-open and unchanged.
- Full regression: 108 passed. See `FINDTEXT_UPSTREAM_NATIVE_ADAPTER_20260904.md`.


## findtextCenterNet platform-native runtime v2.6

- 按上游 `run_ocr.py` 原生后端优先级适配：macOS 默认 CoreML；Windows/Linux 默认量化 ONNX；已有完整 CoreML/ONNX/Torch 任一后端直接复用。
- macOS 默认下载量从 Torch 的约 1422 MiB 降到 CoreML 的约 618 MiB，减少约 56.5%；Windows/Linux 默认 ONNX 约 655 MiB，减少约 53.9%。
- 官方 Hugging Face 仓库为 Xet 存储；新增 `huggingface_hub/hf_xet` 自适应并行下载，旧 `.part` 仍保留作失败回退。
- 修复 Hayai v2.1 SigLIP2 processor 完整性误判导致的重复安装弹窗。
- 完整回归：113 passed。


## findtextCenterNet Xet stall failover v2.7

- Xet 不再可以无限阻塞首次 Ruby 环境准备：默认 `45s` 无 HF incomplete 文件大小/mtime 变化即视为停滞。
- 默认单次 Xet 快速尝试上限 `300s`；真正有缓存写入会持续刷新停滞计时。
- Xet 停滞会终止其独立子进程，保留 Hugging Face/Xet 缓存，然后立即进入可续传 HTTP Range 下载；Hugging Face 大文件优先使用系统 curl 强制 HTTP/1.1，Python HTTPS 作为后备。
- 发生一次 Xet 停滞后默认冷却 6 小时，后续模型文件跳过 Xet；可用 `NOVEL_FORMATTER_FINDTEXT_FORCE_XET=1` 强制再次尝试，或 `HF_HUB_DISABLE_XET=1` / `NOVEL_FORMATTER_FINDTEXT_DISABLE_XET=1` 完全禁用。
- `416 Requested Range Not Satisfiable` 若来自 Xet Range 探测只记为非致命诊断，不再当主要故障展示。
- `HF_TOKEN` 仍为可选；若环境中已有 token，Hugging Face 子进程会自然继承，不额外复制或记录 token。

## findtextCenterNet progress-aware HTTP resume v2.8

- 修复真实首次安装中“每次连接都前进几十 MiB，却因为累计 4+4 次断流在 94% 主动判失败”的问题：重试预算现在只统计**连续无新增字节**的失败。
- 只要 `.part` 在本次连接中继续增长，连接断开会立即从新长度续传，且**不消耗失败预算**；默认单传输层保留 64 次连接安全上限，防止异常无限循环。
- Hugging Face 大文件在 Xet 回退后优先系统 `curl --http1.1 -C -`，规避真实线路反复出现的 `curl (92) HTTP/2 stream ... CANCEL`；Python HTTPS 保留为后备。
- 完成但校验失败的文件仍会删除并重新获取；只有网络中断的部分文件才保留续传，不降低模型/压缩包完整性校验。
- 现有 `.part` 可直接复用；因此覆盖升级后，已经下载到 403 MiB/429 MiB 的 TextDetector 不需要从头开始。
- 可用 `NOVEL_FORMATTER_FINDTEXT_HTTP_MAX_CONNECTIONS=64` 调整单传输层最大连接次数。

### 2026-09-04 · findtextCenterNet v2.11 CoreML 尾部恢复

修复 Hugging Face/CoreML 大文件在 100% 后因极小传输尾部（实测 +44 bytes）被判失败并从 0 B 重下的问题。现在仅在裁到官方大小后 SHA-256 完全正确时自动恢复；否则隔离保留原 `.part`，不会静默删除。详见 `FINDTEXT_COREML_TAIL_RECOVERY_V2_11_20260904.md`。

## Ruby locked exchange v2 (2026-09-05)

`导出融合结果与骨架 EPUB` now emits a locked Ruby exchange contract instead of asking an external AI to preserve Ruby markup by convention.

- `04_ruby_overlay.locked.json` freezes findtextCenterNet readings and context anchors.
- `AI_OUTPUT/edited_text.json` is the only editable text channel.
- `framework/structure_skeleton.epub` and the Ruby lock are SHA-bound in `00_manifest.json`.
- `tools/build_final_epub.py` re-attaches only uniquely resolvable Ruby and drops missing/ambiguous bases fail-closed.
- `tools/validate_ruby.py` reopens the final EPUB, verifies plain text, expected safe Ruby pairs, `<ruby>/<rt>` parity, EPUB mimetype rules, and rejects leftover work attributes.
- Aozora Ruby markers and `<ruby>/<rt>` markup are forbidden in the AI-edited plain-text channel.
- Ruby OFF produces a zero-pair lock and the same builder emits a plain EPUB with no OCR-derived Ruby.

The ordinary OCR voter documents remain Ruby-free; this exchange hardening changes only the external publication/rebuild side-channel.

## Ruby base-edit position mapping v3 (2026-09-05)

Locked Ruby now survives a conservative class of **authoritative OCR/AI corrections inside the Ruby base itself** without ever letting findtextCenterNet replace ordinary OCR prose.

- The implementation follows editor-style position mapping rather than global fuzzy search: the old Ruby span is mapped through the old→new text diff, then the mapped target is validated.
- Exact base/context matching still wins first.
- If the authoritative text changed the Ruby base, migration is allowed only when the new mapped base exactly matches `findtext_detected_base` for the immutable reading (or a legacy explicitly trusted candidate).
- A findtext-verified base may bridge one small substitution/insertion/deletion on short bases (up to two bounded edits on long compounds). This covers OCR corrections such as `椅于→椅子`, `縦密→緻密`, and `水姿見→水の姿見`.
- A same-length one-edit string by itself is **not** enough evidence. For example `魔王《まおう》→魔法` is dropped when findtext had observed `魔王`, even though the strings differ by only one glyph.
- `findtext_detected_reading` is immutable. If it disagrees with the locked reading, migration is rejected instead of falling back to weaker evidence.
- Single-character base migration is accepted only with explicit matching findtext base+reading evidence; unverified `王→玉` still fails closed.
- Insertions exactly at a Ruby span boundary are associated with neighbouring prose rather than silently swallowed into the base.
- Multi-line blocks with repeated identical readings no longer let later findtext lines overwrite earlier base evidence; evidence is merged by stable plain-text anchor before new readings are assigned.
- External locked-exchange bundles now embed the same dependency-free `ruby_anchor.py`, hash every generated tool, and record the anchor policy/version in `00_manifest.json`. Builder and validator therefore use the identical migration policy.

Ruby remains a side-channel: ordinary OCR `block.text`, `block.ocr_raw`, crop bytes, voter input and Fusion prose are never changed by this migration layer. If the mapping or evidence is ambiguous, the Ruby is dropped.

## Ruby base edit-span migration v3 (2026-09-05)

Ruby 仍是与普通 OCR 完全隔离的锁定 side-channel，但现在可以安全跟随 **Ruby 底字本身的 OCR 纠错**。实现采用“旧文本位置 → 新文本位置”的 edit-span 映射，再用 findtextCenterNet 已锁定的 `base + reading` 作为独立证据；不是在整段正文里做模糊字符串猜测。

- 先尝试完全不变的 base/context；若 findtext 曾检测到不同底字，则优先映射原 Ruby span，防止正文别处残留的旧错字抢走 reading。
- `縦密 → 緻密`、`椅于 → 椅子` 这类小范围替换可以迁移；findtext 明确检测到完整底字时，`水姿見 → 水の姿見` / `水のの姿見 → 水の姿見` 这种一个字符的缺失/多余也可以迁移。
- 迁移后的目标必须 **精确等于 findtextCenterNet 当时实际检测到的 base**，reading 仍是不可编辑锁定值；`魔王 → 魔法`、`椅子 → 玉座` 等语义换词不会因为“只差一个字”而继承旧 Ruby。
- 如果纠正后的 base 在同一段中重复且 edit map 没有形成唯一局部上下文，直接丢弃该 Ruby；不会用最近位置、旧 offset 或多数猜测。
- 新 annotation 会保存 `base_original / base_previous / base_edit_history / base_migration_kind / findtext_detected_base / findtext_detected_reading`；真实 findtext char box 可用时还保存原页 `findtext_pair_bbox` 作为审计证据。
- 外部 AI 的 Locked Exchange builder 和 Novel Formatter 内部 `apply_ruby_overlay()` 复用同一份无依赖 `engine/ruby_anchor_core.py`，避免“程序内能保留、导出包却丢 Ruby”的双实现漂移。
- Ruby OFF 路径没有变化：不采集 candidate、不准备/运行 findtext、不执行 edit-span Ruby 迁移，也不会生成 OCR-derived `<ruby>/<rt>`。

该策略只迁移 Ruby metadata；任何情况下都不允许 findtext 或 Ruby 逻辑改写 `block.text` / `ocr_raw` / OCR 投票结果。


### Ruby PositionMap v4: repeated-base identity guard

Ruby/furigana remains a findtextCenterNet-only side channel and never participates in ordinary OCR voting. v4 strengthens post-AI reattachment for repeated homographs:

- a Ruby anchor is identified by its old `source_offset`/span plus edit-map continuity, not by "the spelling is unique now";
- deleting one of two identical bases cannot move its reading to the surviving occurrence;
- an unchanged base moved by edits is re-anchored to fresh offset/context metadata;
- AI-created duplicate phrases are accepted only when the original occurrence remains uniquely identifiable; otherwise Ruby fails closed;
- target spans are reserved as annotations are assigned, so two source readings cannot independently claim the same target before final rendering;
- Locked Exchange builder and validator embed the exact same `engine/ruby_anchor_core.py` policy (`ruby_anchor_policy_version = 4`).

The rule is intentionally conservative: uncertainty drops Ruby while leaving authoritative prose untouched.

### 界面主题与语言

设置 → 常规设置 → 界面与显示中可切换浅色 / 深色主题，以及简体中文 / 日本語 / English。选择后立即预览，点击“保存设置”后写入本机 `QSettings` 并在下次启动时恢复。

主题与语言层只处理 Qt 界面控件、标签、页签、按钮、提示和对话框，不修改 OCR 引擎 ID、模型配置、文档正文、术语、项目文件或 EPUB 数据。OCR / Formatter / OCR 校对 / EPUB 的业务信号和数据链保持不变。

语言切换覆盖主导航、设置、页面管理、OCR、Formatter、OCR 对比、图文对照、EPUB 以及这些工作区的静态说明/工具提示。动态计数与常用状态使用安全的本地规则转换；OCR/AI 原始诊断日志继续保留原始文本，避免因为本地化而改写错误信息或模型输出。

## AI 图文处理（2026-09-20）

新增独立的 **AI 图文处理** 工作区。它不是传统 OCR 的第四个必跑步骤，也不会自动进入 OCR 对比或 Formatter；适合直接让具备图片理解能力的多模态模型完成书页读取、阅读顺序恢复、段落/对白/章节结构化，并可在同一次视觉请求中同时生成译文。

### 输入与页面管理

- 输入仍由现有 **页面管理** 统一接收，因此 PDF、单张图片、多张图片和图片文件夹使用同一条流程。
- AI 工作区读取页面管理中**已确认**的页面分类。封面、彩插、纯插图、空白等已确认资源页不会发送给 API；它们在 EPUB 阶段从原始页面资源重新插回。
- 自动建议但尚未确认的页面类型不会静默阻止识别，避免页面误分类造成正文缺失。
- 页面顺序、原图路径和 EPUB 资源位置由 Novel Formatter 掌握，AI 不负责重新猜测整本书的资产结构。

### 两种任务

- **仅提取原文**：图片 → 结构化日文正文 → 可直接交给 EPUB Builder。
- **提取原文 + 翻译**：第一次视觉请求同时返回日文原文与目标语言初译，避免为翻译再次上传整书图片。译文块绑定源文 block ID 与 SHA-256；缺失译文不会丢掉原文，而会显式标记 `translation_missing`。

### Token / 质量策略

- 默认“均衡”档把正文页最长边控制在约 1920 px；“快速”为 1600 px，“出版”为 2560 px。
- 只裁除明确的大面积纯白页边，不做破坏性锐化或字形变形。
- 默认每次处理连续 2 页，减少重复提示词开销并保留跨页语境；如果模型连续漏回/破坏批次结构，会自动把该批次拆小后重试，直到页级身份可验证，不因一页异常丢掉另一页。
- 模型只为真正看不清的局部返回归一化 bbox；第二遍仅上传这些高分辨率小区域，不重复上传整页。均衡/出版模式还会只针对“整页首轮空结果”做一次高清遗漏保护。
- 对“并非空页、但文本量相对前后两页异常偏少”的强异常页，仅追加一次高清整页漏文审计；只有新结果明显补回正文且仍覆盖大部分首轮文字时才替换，避免把二审幻觉当成更完整的正文。
- 页尾正文没有明确结束符、下一页又以普通正文开头时，先用**纯文本**模型判断是否属于同一跨页段落；确认后按原字串拼接，不让模型重新改写原文，也不重新上传页面图片。
- 相邻页若出现完全相同的长边界文本，只对这一异常边界追加一次定向视觉核验；普通边界不二审。
- 提取+翻译时若少量译文缺失/明显无效，优先用已锁定的日文做纯文本补译，不为补译重新发送页面图片。
- 预处理图片与 AI 响应按源图、模型、提示词和参数做内容哈希缓存；同一未变化图像的编码 payload 也在运行内复用。断点续跑时可直接复用。API Key 不写入该缓存。
- 首轮、局部复核和遗漏保护都绑定源图 SHA-256；同一路径图片在任务中被替换时拒绝把旧视觉结果写回。处理完成到点击 EPUB 之间还会再核对一次页面指纹，防止外部替换图片后把旧文字绑定到新页面。
- 完成后执行零 token 成书预检：正文页经过救援后仍无文字会阻止直接导出；出版档仍有低置信/待视觉复核内容时也会先阻止直接 EPUB。译文缺失继续单独阻止译文 EPUB，不影响已经确认的日文原文。
- 使用量按视觉处理、纯文本跨页检查、纯文本补译分层记录，界面同时显示总 token 与分层 token，便于判断真正的视觉成本。
- 瞬时网络故障采用有限指数退避；明确的“不支持图片”能力错误直接停止，不用无意义的重试消耗请求。

### 多模态模型

AI 图文处理不通过模型名称白名单判断“是否支持图片”。它复用 AI 设置中的 Provider / Model / Base URL / API Key，并提供 **测试图片能力** 按钮，以一次缩小图片的真实请求验证所选端点是否接受视觉输入；AI 设置页另有本地 `NF42` 探针，不依赖用户当前是否已导入书页。

当前传输层支持 OpenAI-compatible（包括 OpenAI、OpenRouter、智谱 BigModel 国内、Z.AI 国际、Ollama、自定义接口等）、Anthropic Messages 与 Google Gemini。智谱国内与 Z.AI 国际现在使用独立 Provider/默认 Base URL，避免把 `open.bigmodel.cn` 的国内密钥误送到 `api.z.ai`。AI 设置页还提供一张本地 `NF42` 探针图，按与 AI 图文工作区完全相同的 Base64 Data URL 路径做真实图片能力验证。具体模型是否真正支持图片，以实际 API 请求为准。

### EPUB 输出

AI 返回的是结构化内容而不是 `.epub` 二进制。最终仍由本地 `EPUB Builder` 确定性生成 EPUB3，并在交接前同步页面管理中的封面和插图。这样可以继续复用现有的 OPF / NAV / spine / CSS / mimetype / 图片资源和完整性检查，同时避免让远端模型重新打包或改写原始图片。

典型流程：

```text
PDF / 图片 / 图片文件夹
        ↓
页面管理（确认正文 / 封面 / 插图等）
        ↓
AI 图文处理（仅提取 / 提取+翻译）
        ↓
AI 疑难区域按需二次复核
        ↓
原文 EPUB / 译文 EPUB
```

传统 `多模型 OCR → OCR 对比 → Formatter → EPUB` 流程保持不变，可继续用于离线处理、独立证据比较或人工精校。


## AI 裁决包 V5

新的分工式多模型 OCR 默认使用 V5 多模型分歧裁决包。V5 保留逐列/整页/全列/分歧复核角色身份，并明确区分真实 OCR 证据与结构 seed；详见 `AI_ADJUDICATION_V5_20260927.md`。

## Project Artifact Pipeline（V5.6）

项目工作区现在额外提供内容寻址 Artifact DAG、OCR Stage Cache 和模型级 Checkpoint Resume。相同页面内容、OCR 参数和实现版本可直接复用已完成模型结果；页面、参数或相关实现代码变化会自动使旧缓存失效。最终 EPUB 会先经过内置 EPUB3/竖排质量闸门，再进入可选 EPUBCheck。

### TEI P5 / W3C EPUB publication bridge（Phase38）

Formatter 现可导入/导出 TEI P5 XML。页面与 OCR 坐标映射为 `facsimile/surface/zone`，最终出版正文与 OCR 原始证据分离，证据通过 `standOff` 保存；`xenoData` 同时保留 Novel Formatter 的完整 UnifiedDocument JSON 快照，因此本程序导出的 TEI 可无损往返，第三方 TEI 工具仍可读取标准层。

EPUB 生成继续以稳定 **EPUB 3.3** 为发布目标，并按 W3C OCF/Packages/Navigation 约束强化内置质量门：修复 `container.xml` OCF namespace，补齐 `dcterms:modified`，XHTML 语言从文档 metadata 继承，不再硬编码 `ja`。EPUBCheck 若不存在会明确显示“未运行”；若实际执行并报告 ERROR/FATAL，则构建失败。EPUBCheck 5.4+ 的 EPUB 3.4 Candidate Recommendation 检查只作为前向兼容观察，不会把出版目标静默升级到草案规范。详见 `PHASE38_TEI_W3C_EPUB_ABSORPTION_AUDIT_20261004.md`。

#### Phase39–40 publication hardening（Phase41 吸收）

在保留 Phase38 Ruby edge-speck hardened OCR 几何/缓存合同的前提下，Phase41 吸收 Phase39–40 的出版层强化：TEI 公共层完整性与标准书目信息往返、EPUB 3.3 ISBN/系列/卷次元数据，以及 Unicode 路径、manifest/spine、Navigation/landmarks、NCX UID、cover-image、ONIX ISBN 类型等 EPUBCheck-parity 质量门。ISBN-10/ISBN-13 分别使用 ONIX Code List 5 的 `02`/`15`。出版目标仍固定为 EPUB 3.3；EPUBCheck 5.4/EPUB 3.4 只作前向兼容信号。详见 `PHASE41_PHASE40_EPUB_PARITY_RUBY_EDGE_ABSORPTION_AUDIT_20261004.md`。

可使用 `tools/ocr_golden_benchmark.py` 对固定人工真值与 OCR 输出计算 CER、Exact Match、替换/删除/插入错误，便于后续对不同版本与 OCR 引擎做可重复 A/B。

> 当前 OCR checkpoint 为模型/阶段级：多模型流程可以从已完成模型继续；单个 OCR adapter 内部的逐页 checkpoint 尚未统一实现。


## Smart OCR Router / Segment Cache / Active Review（V5.7）

分工式多模型 OCR 继续保留完整的角色能力，但 Phase28 的**速度优先默认拓扑**改为：Hayai OCR 512 作为逐列主模型，NDLOCR-Lite 作为整页主模型，全列主模型默认留空，48px AR 只处理 Hayai/NDL 的真实分歧列。历史 `sentence` 持久化键仅作为“全列主模型”的兼容键保留，不再生成合并整句图。用户仍可手动把 48px 放入全列槽做 A/B；默认不再为已经一致的普通正文重复扫描全书物理列。

项目级 `Segment Cache` 位于模型 Stage Cache 下方，按 OCR 输入图像 SHA-256、引擎、操作类型和运行实现指纹缓存单列/整句结果。修改一页不会再必然让其它未变化列和句重新 OCR；失败结果不会写入缓存。OCR 对比增加 `高风险优先`，只调整人工待判断顺序，不改变正文顺序、候选文本或裁决权威。详见 `OCR_ROUTER_SEGMENT_CACHE_ACTIVE_REVIEW_V57_20260928.md`。

## Workspace Compact Storage（2026-09-30）

工作区的大型 OCR / stage / adjudication / Stage Cache JSON 现在使用 `revisions/objects` 下的 SHA-256 内容寻址 gzip 对象，逻辑路径保留为轻量引用；OCR Segment Cache 改为 SQLite，避免数千个小 JSON 的文件系统分配浪费。旧工作区保持兼容，可在左侧独立“工作区”页面点击“优化存储”无损迁移。真实 416 页项目由约 293 MiB logical / 320 MiB filesystem 使用量降至约 94 MiB logical / 97 MiB filesystem 使用量，迁移前后 OCR、stage、裁决和 cache 内容逐哈希一致。详见 `WORKSPACE_STORAGE_OPTIMIZATION_AUDIT_20260930.md`。

## Role4 Final — GPT Quick + Common-Mode Final Audit (2026-10-01)

- Current Phase28 default topology is speed-first: Hayai 512 column main + NDLOCR-Lite page main + empty full-column slot + 48px disagreement-only review. Full-column remains available for manual A/B diagnostics.
- Every true conflict remains in the GPT queue; seeded/copy-forward slots never count as independent OCR evidence.
- Export also creates a compact `_GPT.zip`: GPT edits only `AI_OUTPUT/answers.jsonl`, then runs `python adjudicate.py finish` to produce `AI_IMPORT.zip`.
- Locally agreed rows are no longer assumed infallible. A bounded Common-Mode Final Audit re-opens only high-risk visual/name/structure outliers; it never auto-corrects from corpus frequency.
- The current real-book replay produced 1,172 true conflicts + 94 Common-Mode risks = 1,266 GPT tasks, while keeping the other local-consensus rows out of the AI workload.
- See `ROLE4_FINAL_GPT_COMMON_MODE_AUDIT_20261001.md` for the full real-book benchmark and round-trip validation.
