# OCR 分列字形批量裁决 V5.5

本分支是供原稿盲审和工程测试使用的**离线实验原型**。
不自动修改任何原 OCR 文本，不触发 GitHub Actions，不下载 OCR 模型。
绝不把“同像素”直接等同于正确 Unicode 字码。

## 与已有 V5.3 / V5.4 结果的关系

- V5.3 1036_slot_audit.csv 保存任务 ID、OCR 字符偏移、物理列、像素哈希与拒判理由，但**没有可直接重建字框的四个绝对坐标**。
- V5.4 的 67 张原始分列输入可用于重新核实字符格，并不是 67 次已完成的新 48px 识别。
- 本版本从原 V5 的候选文本重新提取替换位置，不相信旧记录的字符索引。
- 相同原图和坐标被引用多次，不算多个独立重复字形。
- 组级审核完成后，只能输出整句草稿；不得直接标记 AI_IMPORT 已通过。

## 文件

1. engine/ocr_glyph_batch_v55.py — 真实图像来源 SHA、物理字框哈希、审核传播、全差异通过后组装句子。
2. adapters/vertical_glyph_grid_v55.py — 物理竖列投影建议字格，不自动标记坐标已验证。
3. engine/ocr_glyph_grid_alignment_v55.py — 漏字或多字情况下的动态对齐；验证两个独立 OCR 都指向同一格。
4. scripts/ocr_glyph_v53_to_v55.py — 原 V5 ZIP 与 V5.3 审计的转换器。
5. scripts/ocr_glyph_batch_v55.py — 字形审核模板、审核结果应用及只读草稿输出。
6. tests/test_ocr_glyph_batch_v55.py — 24 个针对性测试。
7. tests/test_ocr_glyph_v53_bridge.py — 7 个转换及异常保护测试。
8. tests/test_ocr_glyph_grid_alignment_v55.py — 8 个动态对齐测试。

参考项目：[Kraken OCR](https://github.com/mittagessen/kraken) 的字符位置设计、
[Calamari OCR](https://github.com/Calamari-OCR/calamari) 的序列对齐思路，以及
[manga-ocr-yomitan](https://github.com/peinniku/manga-ocr-yomitan) 一类油猴 OCR
工具的原图坐标绑定、局部缓存及复用思想；没有复制第三方源代码。

## 使用方法

以下命令均从 Novel-formatter 仓库根目录执行。

**1. 将原 V5 包和 V5.3 审计转换成待核验数据：**

    python scripts/ocr_glyph_v53_to_v55.py \
      --package "/path/to/原始V5裁决包.zip" \
      --audit "/path/to/1036_slot_audit.csv" \
      --out "V55_INPUT"

主要输出：

- V5_EDITABLE_CONFLICTS.jsonl — 可用的原始文字分歧任务。
- GLYPH_JOBS_NEED_COORDINATES.jsonl — 字位审核任务，但缺坐标。
- REJECTED_OR_CHANGED_SLOTS.jsonl — V5.3 的旧偏移与原 V5 候选不匹配者。
- SUMMARY_NEEDS_ORIGINAL_BBOX.json — 可核查的实际数量。

特别注意：转换器会让所有转换出的 box 保持空数组，image_path 为空，
position_verified 为 false。这不是 BUG，而是旧 V5.3 CSV 缺少可靠坐标的证明。
任何开发者都不能凭 source_page、col_id、像素 SHA 或 anchor_score
擅自为旧任务填入原图 bbox。

**2. 用真实图片与邻字锚点定位：**

输入图片可以是原始扫描整页或可信的单列原图，但 bbox 必须采用该文件
本身的坐标系 [x0, y0, x1, y1]，不能把整页坐标混用到分列图。
如果使用 V5.4 的单列 PNG，需核对文件 SHA 与其 V5.4 crop_sha256。
必须核实目标字符、前后确认字、列顺序、Ruby，以及来源页。

仅在完成真实原图确认后设置 position_verified=true。自动建议的字格
或单个 OCR 模型给出的索引永远不是已核准原图定位。

**3. 生成组级审核清单：**

    python scripts/ocr_glyph_batch_v55.py \
      --conflicts V55_INPUT/V5_EDITABLE_CONFLICTS.jsonl \
      --glyph-jobs V55_INPUT/GLYPH_JOBS_NEED_COORDINATES.jsonl \
      --out V55_REVIEW

若尚未填写真实坐标，应得到拒绝清单而非大量“已解决”任务。核查：

- GLYPH_REJECTED_JOBS.jsonl — 缺失、不正确或未确认的图像输入。
- GROUPS_UNAPPROVED.json — 原像素相同且物理位置不同的字组。
- GROUP_DECISIONS_TEMPLATE.json — 默认全部 approved=false。
- SUMMARY.json — 只读统计。

**4. 审核原图、日语语境及正确字码后填写组级模板：**

    python scripts/ocr_glyph_batch_v55.py \
      --conflicts V55_INPUT/V5_EDITABLE_CONFLICTS.jsonl \
      --glyph-jobs V55_INPUT/GLYPH_JOBS_NEED_COORDINATES.jsonl \
      --decisions V55_REVIEW/GROUP_DECISIONS_TEMPLATE.json \
      --out V55_REVIEWED

结果 V5_ANSWERS_DRAFT_NOT_IMPORTABLE.jsonl 是待验收草稿，
不是兼容导回的 AI_IMPORT.zip；正式导回前必须使用原包 adjudicate.py
校验，并重新盲查原图、错误传播及最终 EPUB。

### 动态对齐安全约束

- 至少两个经扫描图核实、且不含目标字符本身的独立视觉锚点。
- 分别对两个实际执行过的 OCR 候选计算所有最低代价对齐。
- 只有目标在所有最优对齐中唯一落于同一物理字格，才保留候选。
- OCR seeded/reused 文本不能算独立第二票。
- 字格对齐只建议坐标；不能凭对齐直接推出 Unicode 正字。
- 对「ニ／二」一类同形 Unicode 歧义，必须额外提供整书编辑规范。

## 测试命令

    python -m unittest discover -s tests -p 'test_ocr_glyph_batch_v55.py' -v
    python -m unittest discover -s tests -p 'test_ocr_glyph_v53_bridge.py' -v
    python -m unittest discover -s tests -p 'test_ocr_glyph_grid_alignment_v55.py' -v

共 24 + 7 + 8 = 39 个已编写的针对性测试。**代码已提交不等于测试 PASS。**
本窗口执行容器持续报错，所以尚未成功运行以上测试，也没有重新得到
整书真实覆盖率。不要将以前的 506 / 1036 字位、51 组、421 个候选
算成当前 V5.5 的已完成批量裁决。

## 明确的下一阶段验收门槛（不是已达成成绩）

- 可信字位定位至少 829 / 1036（80%）。
- 原包全部 1259 个待审任务中至少 1008 个完整可核查裁决（80%）。
- 每处传播能追溯到原图 SHA、坐标、原 OCR 候选 SHA、审核记录。
- 用同一套原稿独立比较纯 AI、V5.5 混合方案的实际 tokens、耗时与错字。
- 未满足要求的任务保持待审，不能为了提高完成率把建议写成已批准结果。
