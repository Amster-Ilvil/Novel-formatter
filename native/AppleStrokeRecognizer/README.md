# Apple PKStrokeRecognizer + Vision Candidate Bridge

此目录包含 Novel Formatter 的原生 Apple 桥接，统一提供：

1. `PKStrokeRecognizer` 严格单字手写识别；
2. Apple Vision 单字 Top-N 候选；
3. 原生手写测试与自动轨迹预览面板。

## 自动手写识别

自动模式每次只提交一个字：

```text
一个单字的全部分笔点
→ 稳定 PKStroke
→ 单字 PKDrawing
→ PKStrokeRecognizer
→ 读取首结果
→ 清空并结束当前识别器
→ 下一个字
```

程序不会向系统日语手写键盘窗口注入鼠标事件。

## Vision候选

原生桥接支持：

```text
--vision-candidates
--vision-candidates-batch
```

对每个已切出的单字分别执行两次 `VNRecognizeTextRequest`：

- `usesLanguageCorrection = true`
- `usesLanguageCorrection = false`

每次保留最多20个 `topCandidates`，默认由Python请求Top-10。
整列使用批处理命令在同一个Swift进程内完成。

## 要求

- macOS 27或更高版本
- Xcode 27或更高版本
- macOS 27 SDK
- `PKStrokeRecognizer.supportedLanguages`包含日语

## 编译

```bash
./native/AppleStrokeRecognizer/build.command
```

生成：

```text
native/AppleStrokeRecognizer/AppleStrokeRecognizer.app
native/AppleStrokeRecognizer/bin/apple-stroke-recognizer
```

构建链接Foundation、AppKit、PencilKit和Vision框架。

## 手动测试面板

```bash
open -n native/AppleStrokeRecognizer/AppleStrokeRecognizer.app --args --manual-panel
```

面板可以手动分笔书写、显示苹果实际收到的PKDrawing、查看`recognizedText()`和`indexableContent`，也可以载入最近一次自动轨迹。
