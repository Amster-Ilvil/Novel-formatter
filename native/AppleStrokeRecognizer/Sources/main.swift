import Foundation
import AppKit
import PencilKit
import UniformTypeIdentifiers
import Vision

struct InputPoint: Codable {
    let x: Double
    let y: Double
    let time: Double?
    let width: Double?
    let opacity: Double?
    let force: Double?
}

struct InputStroke: Codable {
    let id: String?
    let glyphIndex: Int?
    let points: [InputPoint]
}

struct RecognitionInput: Codable {
    let preferredLanguages: [String]?
    let canvasWidth: Double?
    let canvasHeight: Double?
    let glyphCount: Int?
    let singleGlyphOnly: Bool?
    let playbackMode: String?
    let pointInterval: Double?
    let strokeGap: Double?
    let updateEveryPoints: Int?
    let debugImages: [String: String]?
    let strokes: [InputStroke]
}

struct DrawingBounds: Codable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

struct BridgeOutput: Codable {
    let bridgeProtocolVersion: Int
    let ok: Bool
    let text: String?
    let recognizedText: String?
    let perGlyphText: String?
    let indexableContent: String?
    let resultSource: String?
    let playbackMode: String?
    let drawingUpdateCount: Int?
    let supportedLanguages: [String]
    let recognizerLanguages: [String]
    let japaneseSupported: Bool
    let recognitionVersion: Int?
    let strokeCount: Int?
    let pointCount: Int?
    let drawingBounds: DrawingBounds?
    let error: String?
}



struct VisionCandidateRequest: Codable {
    let imagePath: String
    let topN: Int?
    let languages: [String]?
}

struct VisionCandidateItem: Codable {
    let text: String
    let confidence: Float
    let rank: Int
    let languageCorrection: Bool
}

struct VisionCandidateResponse: Codable {
    let bridgeProtocolVersion: Int
    let ok: Bool
    let correctedCandidates: [VisionCandidateItem]
    let rawCandidates: [VisionCandidateItem]
    let supportedLanguages: [String]
    let error: String?
}

struct VisionCandidateBatchRequest: Codable {
    let requests: [VisionCandidateRequest]
}

struct VisionCandidateBatchResponse: Codable {
    let bridgeProtocolVersion: Int
    let ok: Bool
    let results: [VisionCandidateResponse]
    let error: String?
}

struct DrawingBundle {
    let drawing: PKDrawing
    let glyphStrokeIDs: [Int: Set<UUID>]
    let pointCount: Int
    let updateCount: Int
}

enum BridgeError: Error, CustomStringConvertible {
    case unsupportedOS
    case japaneseUnavailable([String])
    case emptyInput
    case noValidStrokes
    case multipleGlyphsInSingleMode
    case outputEncodingFailed

    var description: String {
        switch self {
        case .unsupportedOS:
            return "PKStrokeRecognizer requires macOS 27 or later."
        case .japaneseUnavailable(let languages):
            return "Japanese handwriting recognition is unavailable. Supported: \(languages.joined(separator: ", "))"
        case .emptyInput:
            return "No JSON input was provided."
        case .noValidStrokes:
            return "The input did not contain any valid stroke paths."
        case .multipleGlyphsInSingleMode:
            return "Single-glyph mode received strokes from more than one glyph."
        case .outputEncodingFailed:
            return "The bridge response could not be converted to UTF-8 JSON text."
        }
    }
}



@available(macOS 27.0, *)
struct ManualInkPoint {
    let location: CGPoint
    let timeOffset: TimeInterval
}

@available(macOS 27.0, *)
@MainActor
final class ManualStrokeCanvas: NSView {
    var onDrawingChanged: ((Bool) -> Void)?

    private(set) var completedStrokes: [[ManualInkPoint]] = []
    private var currentStroke: [ManualInkPoint] = []
    private var strokeStartTime: TimeInterval = 0

    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer?.backgroundColor = NSColor.white.cgColor
        layer?.borderColor = NSColor.separatorColor.cgColor
        layer?.borderWidth = 1
        layer?.cornerRadius = 8
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func appendPoint(_ location: CGPoint, timestamp: TimeInterval) {
        let clamped = CGPoint(
            x: min(max(0, location.x), bounds.width),
            y: min(max(0, location.y), bounds.height)
        )
        if let last = currentStroke.last {
            let distance = hypot(clamped.x - last.location.x, clamped.y - last.location.y)
            if distance < 0.8 { return }
        }
        currentStroke.append(ManualInkPoint(
            location: clamped,
            timeOffset: max(0, timestamp - strokeStartTime)
        ))
        needsDisplay = true
        onDrawingChanged?(false)
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        strokeStartTime = event.timestamp
        currentStroke = []
        appendPoint(convert(event.locationInWindow, from: nil), timestamp: event.timestamp)
    }

    override func mouseDragged(with event: NSEvent) {
        appendPoint(convert(event.locationInWindow, from: nil), timestamp: event.timestamp)
    }

    override func mouseUp(with event: NSEvent) {
        appendPoint(convert(event.locationInWindow, from: nil), timestamp: event.timestamp)
        if currentStroke.count == 1, let only = currentStroke.first {
            currentStroke.append(ManualInkPoint(
                location: CGPoint(x: only.location.x + 0.8, y: only.location.y + 0.8),
                timeOffset: only.timeOffset + 0.012
            ))
        }
        if currentStroke.count >= 2 {
            completedStrokes.append(currentStroke)
        }
        currentStroke = []
        needsDisplay = true
        onDrawingChanged?(true)
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        NSColor.white.setFill()
        NSBezierPath(rect: bounds).fill()

        NSColor(calibratedWhite: 0.94, alpha: 1).setStroke()
        let grid = NSBezierPath()
        var x: CGFloat = 40
        while x < bounds.width {
            grid.move(to: CGPoint(x: x, y: 0))
            grid.line(to: CGPoint(x: x, y: bounds.height))
            x += 40
        }
        var y: CGFloat = 40
        while y < bounds.height {
            grid.move(to: CGPoint(x: 0, y: y))
            grid.line(to: CGPoint(x: bounds.width, y: y))
            y += 40
        }
        grid.lineWidth = 0.5
        grid.stroke()

        for stroke in completedStrokes {
            drawStroke(stroke, color: .black)
        }
        if !currentStroke.isEmpty {
            drawStroke(currentStroke, color: .systemBlue)
        }
    }

    private func drawStroke(_ stroke: [ManualInkPoint], color: NSColor) {
        guard let first = stroke.first else { return }
        let path = NSBezierPath()
        path.move(to: first.location)
        for point in stroke.dropFirst() {
            path.line(to: point.location)
        }
        path.lineWidth = 3.2
        path.lineCapStyle = .round
        path.lineJoinStyle = .round
        color.setStroke()
        path.stroke()
    }

    func clearDrawing() {
        completedStrokes = []
        currentStroke = []
        needsDisplay = true
        onDrawingChanged?(true)
    }

    func undoLastStroke() {
        if !completedStrokes.isEmpty {
            completedStrokes.removeLast()
        }
        currentStroke = []
        needsDisplay = true
        onDrawingChanged?(true)
    }

    func loadSimpleHorizontalTest() {
        let y = bounds.height * 0.50
        let start = bounds.width * 0.24
        let end = bounds.width * 0.76
        var points: [ManualInkPoint] = []
        let count = 36
        for index in 0..<count {
            let fraction = CGFloat(index) / CGFloat(count - 1)
            points.append(ManualInkPoint(
                location: CGPoint(x: start + (end - start) * fraction, y: y),
                timeOffset: Double(index) * 0.012
            ))
        }
        completedStrokes = [points]
        currentStroke = []
        needsDisplay = true
        onDrawingChanged?(true)
    }

    func pointCount() -> Int {
        completedStrokes.reduce(0) { $0 + $1.count } + currentStroke.count
    }

    func makePKDrawing(includeCurrent: Bool = true) -> PKDrawing {
        let ink = PKInk(.pen, color: NSColor.black)
        let baseDate = Date()
        var source = completedStrokes
        if includeCurrent, currentStroke.count >= 2 {
            source.append(currentStroke)
        }
        var strokes: [PKStroke] = []
        for (strokeIndex, manualStroke) in source.enumerated() where manualStroke.count >= 2 {
            var points: [PKStrokePoint] = []
            points.reserveCapacity(manualStroke.count)
            var lastTime = 0.0
            for (pointIndex, manualPoint) in manualStroke.enumerated() {
                let time = pointIndex == 0 ? 0.0 : max(lastTime + 0.001, manualPoint.timeOffset)
                lastTime = time
                points.append(PKStrokePoint(
                    location: manualPoint.location,
                    timeOffset: time,
                    size: CGSize(width: 3.2, height: 3.2),
                    opacity: 1.0,
                    force: 0.55,
                    azimuth: 0,
                    altitude: .pi / 2
                ))
            }
            let path = PKStrokePath(
                controlPoints: points,
                creationDate: baseDate.addingTimeInterval(Double(strokeIndex) * 0.080)
            )
            strokes.append(PKStroke(ink: ink, path: path, transform: .identity, mask: nil))
        }
        return PKDrawing(strokes: strokes)
    }
}

@available(macOS 27.0, *)
@MainActor
final class ManualRecognitionPanel: NSObject, NSWindowDelegate {
    private let recognizer: PKStrokeRecognizer
    private let languageIdentifier: String
    private let canvas = ManualStrokeCanvas(frame: .zero)
    private let preview = NSImageView(frame: .zero)
    private let debugPreview = NSImageView(frame: .zero)
    private let resultView = NSTextView(frame: .zero)
    private let indexableView = NSTextView(frame: .zero)
    private let statusLabel = NSTextField(labelWithString: "")
    private let liveCheckbox = NSButton(checkboxWithTitle: "每次抬笔后自动识别", target: nil, action: nil)
    private var recognitionGeneration = 0
    private var window: NSWindow!
    private var loadedDrawing: PKDrawing?
    private var loadedGlyphStrokeIDs: [Int: Set<UUID>] = [:]
    private var loadedPointCount: Int = 0
    private var loadedCanvasSize = CGSize(width: 842, height: 595)
    private var loadedPayloadURL: URL?

    init(language: Locale.Language, payloadURL: URL? = nil) {
        self.languageIdentifier = language.minimalIdentifier
        self.recognizer = PKStrokeRecognizer(preferredLanguages: [language])
        super.init()
        buildWindow()
        if let payloadURL {
            loadAutomaticPayload(from: payloadURL, recognizeAfterLoad: true)
        }
    }

    func show() {
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        refreshPreview()
    }

    private func makeButton(_ title: String, action: Selector) -> NSButton {
        let button = NSButton(title: title, target: self, action: action)
        button.bezelStyle = .rounded
        return button
    }

    private func configuredTextView(fontSize: CGFloat) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.borderType = .bezelBorder
        let text = NSTextView()
        text.isEditable = false
        text.isSelectable = true
        text.font = NSFont.systemFont(ofSize: fontSize)
        text.textContainerInset = NSSize(width: 8, height: 8)
        scroll.documentView = text
        if fontSize >= 20 {
            resultView.font = text.font
        }
        return scroll
    }

    private func buildWindow() {
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 900),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "PKStrokeRecognizer 日语手写测试"
        window.center()
        window.minSize = NSSize(width: 980, height: 760)
        window.delegate = self

        let root = NSView()
        root.translatesAutoresizingMaskIntoConstraints = false
        window.contentView = root

        let title = NSTextField(labelWithString: "PKStrokeRecognizer 手动输入与 PKDrawing 预览")
        title.font = NSFont.boldSystemFont(ofSize: 17)
        let subtitle = NSTextField(wrappingLabelWithString:
            "左侧可手动分笔书写；也可载入程序最近生成的自动轨迹 JSON。右侧由 PKDrawing.image(from:scale:) 直接渲染。点击“复制并返回 Novel Formatter”后，结果会自动写入当前图文校对文本框。"
        )
        subtitle.textColor = .secondaryLabelColor

        canvas.translatesAutoresizingMaskIntoConstraints = false
        canvas.onDrawingChanged = { [weak self] strokeEnded in
            guard let self else { return }
            self.refreshPreview()
            if strokeEnded, self.liveCheckbox.state == .on {
                self.recognizeDrawing()
            }
        }

        preview.translatesAutoresizingMaskIntoConstraints = false
        preview.imageAlignment = .alignCenter
        preview.imageScaling = .scaleProportionallyDown
        preview.wantsLayer = true
        preview.layer?.backgroundColor = NSColor.white.cgColor
        preview.layer?.borderColor = NSColor.separatorColor.cgColor
        preview.layer?.borderWidth = 1
        preview.layer?.cornerRadius = 8

        debugPreview.translatesAutoresizingMaskIntoConstraints = false
        debugPreview.imageAlignment = .alignCenter
        debugPreview.imageScaling = .scaleProportionallyDown
        debugPreview.wantsLayer = true
        debugPreview.layer?.backgroundColor = NSColor.white.cgColor
        debugPreview.layer?.borderColor = NSColor.separatorColor.cgColor
        debugPreview.layer?.borderWidth = 1
        debugPreview.layer?.cornerRadius = 8

        let canvasTitle = NSTextField(labelWithString: "① 手动分笔书写")
        canvasTitle.font = NSFont.boldSystemFont(ofSize: 13)
        let debugTitle = NSTextField(labelWithString: "② 自动字形四阶段：原图｜保守掩膜｜高分辨率二值｜最终轨迹")
        debugTitle.font = NSFont.boldSystemFont(ofSize: 13)
        let previewTitle = NSTextField(labelWithString: "③ 苹果实际收到的 PKDrawing")
        previewTitle.font = NSFont.boldSystemFont(ofSize: 13)

        let left = NSStackView(views: [canvasTitle, canvas])
        left.orientation = .vertical
        left.alignment = .leading
        left.spacing = 7
        left.translatesAutoresizingMaskIntoConstraints = false
        canvas.widthAnchor.constraint(equalTo: left.widthAnchor).isActive = true

        let resultScroll = NSScrollView()
        resultScroll.hasVerticalScroller = true
        resultScroll.borderType = .bezelBorder
        resultView.isEditable = false
        resultView.isSelectable = true
        resultView.isHorizontallyResizable = false
        resultView.isVerticallyResizable = true
        resultView.autoresizingMask = [.width]
        resultView.font = NSFont.systemFont(ofSize: 30, weight: .medium)
        resultView.textContainerInset = NSSize(width: 10, height: 10)
        resultView.textContainer?.widthTracksTextView = true
        resultView.textContainer?.containerSize = NSSize(width: 430, height: CGFloat.greatestFiniteMagnitude)
        resultView.frame = NSRect(x: 0, y: 0, width: 430, height: 115)
        resultScroll.documentView = resultView

        let indexScroll = NSScrollView()
        indexScroll.hasVerticalScroller = true
        indexScroll.borderType = .bezelBorder
        indexableView.isEditable = false
        indexableView.isSelectable = true
        indexableView.isHorizontallyResizable = false
        indexableView.isVerticallyResizable = true
        indexableView.autoresizingMask = [.width]
        indexableView.font = NSFont.systemFont(ofSize: 13)
        indexableView.textContainerInset = NSSize(width: 8, height: 8)
        indexableView.textContainer?.widthTracksTextView = true
        indexableView.textContainer?.containerSize = NSSize(width: 430, height: CGFloat.greatestFiniteMagnitude)
        indexableView.frame = NSRect(x: 0, y: 0, width: 430, height: 72)
        indexScroll.documentView = indexableView

        let resultTitle = NSTextField(labelWithString: "④ recognizedText() 首结果")
        resultTitle.font = NSFont.boldSystemFont(ofSize: 13)
        let indexTitle = NSTextField(labelWithString: "indexableContent（辅助观察）")
        indexTitle.font = NSFont.systemFont(ofSize: 12, weight: .semibold)

        statusLabel.lineBreakMode = .byWordWrapping
        statusLabel.maximumNumberOfLines = 3
        statusLabel.textColor = .secondaryLabelColor
        liveCheckbox.target = self
        liveCheckbox.action = #selector(liveModeChanged)
        liveCheckbox.state = .on

        let recognitionButtons = NSStackView(views: [
            makeButton("立即识别", action: #selector(recognizePressed)),
            makeButton("载入最新自动轨迹", action: #selector(loadLatestPayloadPressed)),
            makeButton("选择轨迹 JSON", action: #selector(choosePayloadPressed)),
            makeButton("返回手动画板", action: #selector(returnToManualPressed))
        ])
        recognitionButtons.orientation = .horizontal
        recognitionButtons.spacing = 7
        recognitionButtons.distribution = .gravityAreas

        let editButtons = NSStackView(views: [
            makeButton("撤销一笔", action: #selector(undoPressed)),
            makeButton("清空", action: #selector(clearPressed)),
            makeButton("载入「一」测试", action: #selector(loadTestPressed)),
            makeButton("复制结果", action: #selector(copyPressed)),
            makeButton("复制并返回 Novel Formatter", action: #selector(copyAndClosePressed))
        ])
        editButtons.orientation = .horizontal
        editButtons.spacing = 7
        editButtons.distribution = .gravityAreas

        let right = NSStackView(views: [
            debugTitle, debugPreview, previewTitle, preview, resultTitle, resultScroll,
            indexTitle, indexScroll, liveCheckbox, statusLabel, recognitionButtons, editButtons
        ])
        right.orientation = .vertical
        right.alignment = .leading
        right.spacing = 7
        right.translatesAutoresizingMaskIntoConstraints = false
        debugPreview.widthAnchor.constraint(equalTo: right.widthAnchor).isActive = true
        preview.widthAnchor.constraint(equalTo: right.widthAnchor).isActive = true
        resultScroll.widthAnchor.constraint(equalTo: right.widthAnchor).isActive = true
        indexScroll.widthAnchor.constraint(equalTo: right.widthAnchor).isActive = true

        let columns = NSStackView(views: [left, right])
        columns.orientation = .horizontal
        columns.distribution = .fillEqually
        columns.alignment = .top
        columns.spacing = 14
        columns.translatesAutoresizingMaskIntoConstraints = false

        root.addSubview(title)
        root.addSubview(subtitle)
        root.addSubview(columns)
        title.translatesAutoresizingMaskIntoConstraints = false
        subtitle.translatesAutoresizingMaskIntoConstraints = false

        NSLayoutConstraint.activate([
            title.topAnchor.constraint(equalTo: root.topAnchor, constant: 16),
            title.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: 18),
            title.trailingAnchor.constraint(equalTo: root.trailingAnchor, constant: -18),
            subtitle.topAnchor.constraint(equalTo: title.bottomAnchor, constant: 5),
            subtitle.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            subtitle.trailingAnchor.constraint(equalTo: title.trailingAnchor),
            columns.topAnchor.constraint(equalTo: subtitle.bottomAnchor, constant: 13),
            columns.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: 18),
            columns.trailingAnchor.constraint(equalTo: root.trailingAnchor, constant: -18),
            columns.bottomAnchor.constraint(equalTo: root.bottomAnchor, constant: -18),
            canvas.heightAnchor.constraint(greaterThanOrEqualToConstant: 720),
            debugPreview.heightAnchor.constraint(equalToConstant: 155),
            preview.heightAnchor.constraint(equalToConstant: 225),
            resultScroll.heightAnchor.constraint(equalToConstant: 115),
            indexScroll.heightAnchor.constraint(equalToConstant: 72),
        ])
    }

    private func activeDrawing() -> PKDrawing {
        loadedDrawing ?? canvas.makePKDrawing(includeCurrent: true)
    }

    private func activePointCount() -> Int {
        loadedDrawing == nil ? canvas.pointCount() : loadedPointCount
    }

    private func activeRenderRect() -> CGRect {
        if loadedDrawing != nil {
            return CGRect(origin: .zero, size: loadedCanvasSize)
        }
        return canvas.bounds
    }

    private func refreshPreview() {
        let drawing = activeDrawing()
        preview.image = drawing.image(from: activeRenderRect(), scale: 1.0)
        let bounds = drawing.bounds
        let source = loadedPayloadURL?.lastPathComponent ?? "手动画板"
        statusLabel.stringValue = String(
            format: "来源：%@｜语言：%@｜笔画：%d｜采样点：%d｜PKDrawing bounds：%.1f × %.1f",
            source, languageIdentifier, drawing.strokes.count, activePointCount(), bounds.width, bounds.height
        )
        if drawing.strokes.isEmpty {
            resultView.string = ""
            indexableView.string = ""
        }
    }

    @objc private func recognizePressed() {
        recognizeDrawing()
    }

    @objc private func undoPressed() {
        if loadedDrawing != nil { returnToManual() }
        canvas.undoLastStroke()
    }

    @objc private func clearPressed() {
        recognitionGeneration += 1
        returnToManual()
        canvas.clearDrawing()
        resultView.string = ""
        indexableView.string = ""
    }

    @objc private func loadTestPressed() {
        returnToManual()
        canvas.loadSimpleHorizontalTest()
    }

    private func returnToManual() {
        loadedDrawing = nil
        loadedGlyphStrokeIDs = [:]
        loadedPointCount = 0
        loadedPayloadURL = nil
        debugPreview.image = nil
        refreshPreview()
    }

    @objc private func returnToManualPressed() {
        returnToManual()
    }

    private func latestPayloadURL() -> URL? {
        let fm = FileManager.default
        var roots: [URL] = []
        if let configured = ProcessInfo.processInfo.environment["NOVEL_FORMATTER_ROOT"], !configured.isEmpty {
            roots.append(URL(fileURLWithPath: configured, isDirectory: true))
        }
        roots.append(URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true))
        for root in roots {
            let fixed = root.appendingPathComponent("debug/apple_pkstroke/latest-auto-input.payload.json")
            if fm.fileExists(atPath: fixed.path) { return fixed }
            let folder = root.appendingPathComponent("debug/apple_pkstroke", isDirectory: true)
            guard let items = try? fm.contentsOfDirectory(
                at: folder, includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            ) else { continue }
            let candidates = items.filter { $0.lastPathComponent.hasSuffix(".payload.json") }
                .sorted {
                    let left = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                    let right = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                    return left > right
                }
            if let first = candidates.first { return first }
        }
        return nil
    }

    @objc private func loadLatestPayloadPressed() {
        guard let url = latestPayloadURL() else {
            statusLabel.stringValue = "没有找到自动轨迹。请先运行一次黑像素临摹识别，或点击‘选择轨迹 JSON’。"
            return
        }
        loadAutomaticPayload(from: url, recognizeAfterLoad: true)
    }

    @objc private func choosePayloadPressed() {
        let panel = NSOpenPanel()
        panel.title = "选择自动 PKStroke 轨迹 JSON"
        panel.allowedContentTypes = [.json]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        loadAutomaticPayload(from: url, recognizeAfterLoad: true)
    }

    private func loadAutomaticPayload(from url: URL, recognizeAfterLoad: Bool) {
        do {
            let data = try Data(contentsOf: url)
            let input = try JSONDecoder().decode(RecognitionInput.self, from: data)
            let ink = PKInk(.pen, color: NSColor.black)
            let pointInterval = max(0.002, min(0.050, input.pointInterval ?? 0.010))
            let strokeGap = max(0.010, min(0.300, input.strokeGap ?? 0.065))
            let baseDate = Date()
            var elapsed = 0.0
            var strokes: [PKStroke] = []
            var glyphIDs: [Int: Set<UUID>] = [:]
            var count = 0
            for sourceStroke in input.strokes where sourceStroke.points.count >= 2 {
                let points = AppleStrokeRecognizerCLI.normalizedPoints(sourceStroke, pointInterval: pointInterval)
                guard points.count >= 2 else { continue }
                let stroke = AppleStrokeRecognizerCLI.makeStroke(
                    points: points,
                    creationDate: baseDate.addingTimeInterval(elapsed),
                    ink: ink
                )
                strokes.append(stroke)
                if let glyphIndex = sourceStroke.glyphIndex {
                    glyphIDs[glyphIndex, default: []].insert(stroke.id)
                }
                count += points.count
                elapsed += (points.last?.timeOffset ?? pointInterval * Double(points.count - 1)) + strokeGap
            }
            guard !strokes.isEmpty else {
                statusLabel.stringValue = "所选 JSON 没有有效笔画。"
                return
            }
            loadedDrawing = PKDrawing(strokes: strokes)
            loadedGlyphStrokeIDs = glyphIDs
            loadedPointCount = count
            loadedCanvasSize = CGSize(
                width: max(1, input.canvasWidth ?? 842),
                height: max(1, input.canvasHeight ?? 595)
            )
            loadedPayloadURL = url
            if let comparisonPath = input.debugImages?["comparison"],
               FileManager.default.fileExists(atPath: comparisonPath) {
                debugPreview.image = NSImage(contentsOfFile: comparisonPath)
            } else {
                debugPreview.image = nil
            }
            refreshPreview()
            if recognizeAfterLoad { recognizeDrawing() }
        } catch {
            statusLabel.stringValue = "载入自动轨迹失败：\(error.localizedDescription)"
        }
    }

    private func copyResultToNovelFormatter() -> Bool {
        let value = resultView.string.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else {
            statusLabel.stringValue = "当前没有可复制的手写识别结果。"
            return false
        }
        let marker = NSPasteboard.PasteboardType("com.novelformatter.apple-handwriting")
        NSPasteboard.general.clearContents()
        NSPasteboard.general.declareTypes([.string, marker], owner: nil)
        NSPasteboard.general.setString(value, forType: .string)
        NSPasteboard.general.setString("1", forType: marker)
        statusLabel.stringValue = "已复制结果；Novel Formatter 会自动写入当前图文校对文本框。"
        return true
    }

    @objc private func copyPressed() {
        _ = copyResultToNovelFormatter()
    }

    @objc private func copyAndClosePressed() {
        if copyResultToNovelFormatter() {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                NSApp.terminate(nil)
            }
        }
    }

    @objc private func liveModeChanged() {
        if liveCheckbox.state == .on, !canvas.completedStrokes.isEmpty {
            recognizeDrawing()
        }
    }

    private func recognizeDrawing() {
        let drawing = loadedDrawing ?? canvas.makePKDrawing(includeCurrent: false)
        guard !drawing.strokes.isEmpty else {
            statusLabel.stringValue = "请先在左侧分笔书写。"
            return
        }
        recognitionGeneration += 1
        let generation = recognitionGeneration
        statusLabel.stringValue = "正在调用 PKStrokeRecognizer（\(languageIdentifier)）…"
        let recognizer = self.recognizer
        Task { @MainActor [weak self] in
            guard let self else { return }
            await recognizer.updateDrawing(drawing)
            // The macOS 27 SDK exposes both values as optional strings.
            // Treat nil as an empty result so the manual panel compiles across
            // SDK revisions and can still show a clear “未返回文字” state.
            var text = (await recognizer.recognizedText()) ?? ""
            let indexable = (await recognizer.indexableContent) ?? ""
            if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !self.loadedGlyphStrokeIDs.isEmpty {
                let maxIndex = self.loadedGlyphStrokeIDs.keys.max() ?? -1
                if maxIndex >= 0 {
                    var slots: [String] = []
                    for index in 0...maxIndex {
                        if let ids = self.loadedGlyphStrokeIDs[index], !ids.isEmpty {
                            slots.append((await recognizer.recognizedText(strokeIDs: ids)) ?? "□")
                        } else {
                            slots.append("□")
                        }
                    }
                    let joined = slots.joined()
                    if joined.contains(where: { $0 != "□" }) { text = joined }
                }
            }
            guard generation == self.recognitionGeneration else { return }
            self.resultView.string = text
            self.indexableView.string = indexable
            let bounds = drawing.bounds
            self.statusLabel.stringValue = String(
                format: "识别完成｜语言：%@｜笔画：%d｜采样点：%d｜bounds：%.1f × %.1f｜%@",
                self.languageIdentifier,
                drawing.strokes.count,
                self.activePointCount(),
                bounds.width,
                bounds.height,
                text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "未返回文字" : "已返回首结果"
            )
            self.preview.image = drawing.image(from: self.activeRenderRect(), scale: 1.0)
        }
    }

    func windowWillClose(_ notification: Notification) {
        NSApp.terminate(nil)
    }
}

@main
struct AppleStrokeRecognizerCLI {
    static let protocolVersion = 11

    static func encodeAndPrint<Output: Encodable>(_ output: Output) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        do {
            let data = try encoder.encode(output)
            guard let text = String(data: data, encoding: .utf8) else {
                throw BridgeError.outputEncodingFailed
            }
            FileHandle.standardOutput.write(Data((text + "\n").utf8))
        } catch {
            let message = String(describing: error)
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
                .replacingOccurrences(of: "\n", with: "\\n")
            FileHandle.standardOutput.write(Data(("{\"ok\":false,\"error\":\"Failed to encode bridge output: \(message)\"}\n").utf8))
        }
    }

    static func clean(_ value: String?) -> String? {
        guard let value else { return nil }
        let cleaned = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty ? nil : cleaned
    }

    @available(macOS 27.0, *)
    static func languageStatus() -> ([Locale.Language], [String], Locale.Language?) {
        let values = Array(PKStrokeRecognizer.supportedLanguages)
        let names = values.map { $0.minimalIdentifier }.sorted()
        let japaneseRequest = Locale.Language(identifier: "ja-JP")
        let actualJapanese = values.first { $0.isEquivalent(to: japaneseRequest) }
        return (values, names, actualJapanese)
    }

    @available(macOS 27.0, *)
    static func resolvedPreferredLanguages(
        requestedIdentifiers: [String]?,
        supported: [Locale.Language],
        actualJapanese: Locale.Language?
    ) -> [Locale.Language] {
        let requested = (requestedIdentifiers ?? ["ja-JP"]).map { Locale.Language(identifier: $0) }
        var result: [Locale.Language] = []
        for request in requested {
            if let actual = supported.first(where: { $0.isEquivalent(to: request) }),
               !result.contains(where: { $0.isEquivalent(to: actual) }) {
                result.append(actual)
            }
        }
        if result.isEmpty, let actualJapanese {
            result.append(actualJapanese)
        }
        return result
    }

    static func normalizedPoints(
        _ sourceStroke: InputStroke,
        pointInterval: Double
    ) -> [PKStrokePoint] {
        var result: [PKStrokePoint] = []
        result.reserveCapacity(sourceStroke.points.count)
        var lastTime = 0.0
        for (index, sourcePoint) in sourceStroke.points.enumerated() {
            let proposed = sourcePoint.time ?? (Double(index) * pointInterval)
            let timeOffset = index == 0 ? 0.0 : max(lastTime + 0.001, proposed)
            lastTime = timeOffset
            let width = max(1.2, min(10.0, sourcePoint.width ?? 3.2))
            result.append(PKStrokePoint(
                location: CGPoint(x: sourcePoint.x, y: sourcePoint.y),
                timeOffset: timeOffset,
                size: CGSize(width: width, height: width),
                opacity: max(0.1, min(1.0, sourcePoint.opacity ?? 1.0)),
                force: max(0.05, min(1.0, sourcePoint.force ?? 0.55)),
                azimuth: 0,
                altitude: .pi / 2
            ))
        }
        return result
    }

    @available(macOS 27.0, *)
    static func makeStroke(
        points: [PKStrokePoint],
        creationDate: Date,
        ink: PKInk
    ) -> PKStroke {
        let path = PKStrokePath(controlPoints: points, creationDate: creationDate)
        return PKStroke(ink: ink, path: path, transform: .identity, mask: nil)
    }

    /// Replays the point stream inside PencilKit instead of handing the
    /// recognizer a single prebuilt drawing. Each source array is one pen-down
    /// stroke. The current stroke grows point by point; after each update batch,
    /// PKStrokeRecognizer receives the new partial PKDrawing. At stroke end the
    /// pen is lifted, a creation-date gap is inserted, and the next stroke starts.
    @available(macOS 27.0, *)
    static func replayDrawing(
        from input: RecognitionInput,
        recognizer: PKStrokeRecognizer
    ) async throws -> DrawingBundle {
        let ink = PKInk(.pen, color: NSColor.black)
        let pointInterval = max(0.002, min(0.050, input.pointInterval ?? 0.010))
        let strokeGap = max(0.010, min(0.300, input.strokeGap ?? 0.065))
        let updateEvery = max(1, min(12, input.updateEveryPoints ?? 3))
        let baseDate = Date()

        var completedStrokes: [PKStroke] = []
        var glyphStrokeIDs: [Int: Set<UUID>] = [:]
        var pointCount = 0
        var updateCount = 0
        var elapsed = 0.0

        for sourceStroke in input.strokes {
            guard sourceStroke.points.count >= 2 else { continue }
            let allPoints = normalizedPoints(sourceStroke, pointInterval: pointInterval)
            guard allPoints.count >= 2 else { continue }
            let creationDate = baseDate.addingTimeInterval(elapsed)

            // Point-by-point replay. PKStrokePath performs Apple's own spline
            // connection between the supplied control points.
            for endIndex in 2...allPoints.count {
                let shouldUpdate = endIndex == 2 || endIndex == allPoints.count || endIndex % updateEvery == 0
                guard shouldUpdate else { continue }
                let partialStroke = makeStroke(
                    points: Array(allPoints.prefix(endIndex)),
                    creationDate: creationDate,
                    ink: ink
                )
                let partialDrawing = PKDrawing(strokes: completedStrokes + [partialStroke])
                await recognizer.updateDrawing(partialDrawing)
                updateCount += 1
                await Task.yield()
            }

            let finalStroke = makeStroke(points: allPoints, creationDate: creationDate, ink: ink)
            completedStrokes.append(finalStroke)
            if let glyphIndex = sourceStroke.glyphIndex {
                glyphStrokeIDs[glyphIndex, default: []].insert(finalStroke.id)
            }
            pointCount += allPoints.count
            elapsed += (allPoints.last?.timeOffset ?? pointInterval * Double(allPoints.count - 1)) + strokeGap

            // Explicit pen-up boundary: submit the committed stroke collection
            // before beginning the next stroke.
            await recognizer.updateDrawing(PKDrawing(strokes: completedStrokes))
            updateCount += 1
            await Task.yield()
        }

        guard !completedStrokes.isEmpty else { throw BridgeError.noValidStrokes }
        let finalDrawing = PKDrawing(strokes: completedStrokes)
        await recognizer.updateDrawing(finalDrawing)
        updateCount += 1
        return DrawingBundle(
            drawing: finalDrawing,
            glyphStrokeIDs: glyphStrokeIDs,
            pointCount: pointCount,
            updateCount: updateCount
        )
    }

    @available(macOS 27.0, *)
    static func makeDrawingInstantly(
        from input: RecognitionInput,
        recognizer: PKStrokeRecognizer
    ) async throws -> DrawingBundle {
        let ink = PKInk(.pen, color: NSColor.black)
        let pointInterval = max(0.002, min(0.050, input.pointInterval ?? 0.010))
        let strokeGap = max(0.010, min(0.300, input.strokeGap ?? 0.065))
        let baseDate = Date()
        var elapsed = 0.0
        var strokes: [PKStroke] = []
        var glyphStrokeIDs: [Int: Set<UUID>] = [:]
        var pointCount = 0

        for sourceStroke in input.strokes {
            guard sourceStroke.points.count >= 2 else { continue }
            let points = normalizedPoints(sourceStroke, pointInterval: pointInterval)
            guard points.count >= 2 else { continue }
            let stroke = makeStroke(
                points: points,
                creationDate: baseDate.addingTimeInterval(elapsed),
                ink: ink
            )
            strokes.append(stroke)
            if let glyphIndex = sourceStroke.glyphIndex {
                glyphStrokeIDs[glyphIndex, default: []].insert(stroke.id)
            }
            pointCount += points.count
            elapsed += (points.last?.timeOffset ?? pointInterval * Double(points.count - 1)) + strokeGap
        }
        guard !strokes.isEmpty else { throw BridgeError.noValidStrokes }
        let drawing = PKDrawing(strokes: strokes)
        await recognizer.updateDrawing(drawing)
        return DrawingBundle(
            drawing: drawing,
            glyphStrokeIDs: glyphStrokeIDs,
            pointCount: pointCount,
            updateCount: 1
        )
    }


    @MainActor
    static func pumpAppKitRunLoop(for seconds: TimeInterval) {
        // The working manual panel spends its lifetime inside NSApplication.run().
        // A one-shot CLI process must still service the main run loop so
        // PencilKit can publish its asynchronous recognition state.
        RunLoop.main.run(until: Date(timeIntervalSinceNow: max(0.01, seconds)))
    }

    @available(macOS 27.0, *)
    @MainActor
    static func waitForRecognition(
        recognizer: PKStrokeRecognizer,
        bundle: DrawingBundle,
        glyphCount: Int?,
        attempts: Int = 28,
        delayNanoseconds: UInt64 = 90_000_000
    ) async -> (whole: String?, indexable: String?, perGlyph: String?) {
        var latestWhole: String? = nil
        var latestIndexable: String? = nil
        var latestPerGlyph: String? = nil
        let safeAttempts = max(1, attempts)

        for attempt in 0..<safeAttempts {
            latestWhole = clean(await recognizer.recognizedText())
            latestIndexable = clean(await recognizer.indexableContent)

            if latestWhole == nil, !bundle.glyphStrokeIDs.isEmpty {
                let count = max(glyphCount ?? 0, (bundle.glyphStrokeIDs.keys.max() ?? -1) + 1)
                if count > 0 {
                    var chars: [String] = []
                    chars.reserveCapacity(count)
                    for glyphIndex in 0..<count {
                        guard let ids = bundle.glyphStrokeIDs[glyphIndex], !ids.isEmpty else {
                            chars.append("□")
                            continue
                        }
                        chars.append(clean(await recognizer.recognizedText(strokeIDs: ids)) ?? "□")
                    }
                    let joined = chars.joined()
                    latestPerGlyph = joined.contains(where: { $0 != "□" }) ? joined : nil
                }
            }

            if latestWhole != nil || latestPerGlyph != nil || latestIndexable != nil {
                break
            }
            if attempt + 1 < safeAttempts {
                let delaySeconds = Double(delayNanoseconds) / 1_000_000_000.0
                await pumpAppKitRunLoop(for: delaySeconds)
                await Task.yield()
            }
        }
        return (latestWhole, latestIndexable, latestPerGlyph)
    }

    static func visionSupportedLanguages() -> [String] {
        let request = VNRecognizeTextRequest()
        do {
            return try request.supportedRecognitionLanguages().sorted()
        } catch {
            return ["ja-JP"]
        }
    }

    static func cgImage(at path: String) throws -> CGImage {
        let url = URL(fileURLWithPath: path)
        guard let image = NSImage(contentsOf: url),
              let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NSError(domain: "AppleVisionCandidates", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unable to load glyph image: \(path)"])
        }
        return cgImage
    }

    static func runVisionCandidates(
        cgImage: CGImage,
        topN: Int,
        languages: [String],
        languageCorrection: Bool
    ) throws -> [VisionCandidateItem] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = languageCorrection
        request.minimumTextHeight = 0.01
        let supported = Set(visionSupportedLanguages())
        let selected = languages.filter { supported.contains($0) }
        if !selected.isEmpty {
            request.recognitionLanguages = selected
        } else if supported.contains("ja-JP") {
            request.recognitionLanguages = ["ja-JP"]
        }
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])
        guard let observations = request.results, !observations.isEmpty else { return [] }

        // A glyph crop can occasionally produce more than one observation when
        // detached dots or punctuation are present.  Prefer the observation
        // covering the largest area, then the highest first-candidate confidence.
        let observation = observations.max { left, right in
            let leftCandidate = left.topCandidates(1).first?.confidence ?? 0
            let rightCandidate = right.topCandidates(1).first?.confidence ?? 0
            let leftScore = Float(left.boundingBox.width * left.boundingBox.height) + leftCandidate * 0.15
            let rightScore = Float(right.boundingBox.width * right.boundingBox.height) + rightCandidate * 0.15
            return leftScore < rightScore
        }
        guard let observation else { return [] }
        return observation.topCandidates(max(1, min(20, topN))).enumerated().map { index, candidate in
            VisionCandidateItem(
                text: candidate.string,
                confidence: candidate.confidence,
                rank: index + 1,
                languageCorrection: languageCorrection
            )
        }
    }

    static func recognizeVisionCandidates(_ input: VisionCandidateRequest) -> VisionCandidateResponse {
        let supported = visionSupportedLanguages()
        do {
            let image = try cgImage(at: input.imagePath)
            let topN = max(1, min(20, input.topN ?? 10))
            let languages = input.languages ?? ["ja-JP"]
            let corrected = try runVisionCandidates(
                cgImage: image, topN: topN, languages: languages, languageCorrection: true
            )
            let raw = try runVisionCandidates(
                cgImage: image, topN: topN, languages: languages, languageCorrection: false
            )
            return VisionCandidateResponse(
                bridgeProtocolVersion: protocolVersion,
                ok: !corrected.isEmpty || !raw.isEmpty,
                correctedCandidates: corrected,
                rawCandidates: raw,
                supportedLanguages: supported,
                error: (!corrected.isEmpty || !raw.isEmpty) ? nil : "Apple Vision returned no text candidates for the glyph."
            )
        } catch {
            return VisionCandidateResponse(
                bridgeProtocolVersion: protocolVersion,
                ok: false,
                correctedCandidates: [],
                rawCandidates: [],
                supportedLanguages: supported,
                error: String(describing: error)
            )
        }
    }

    @available(macOS 27.0, *)
    static func statusOutput() -> BridgeOutput {
        let (_, languages, actualJapanese) = languageStatus()
        let hasJapanese = actualJapanese != nil
        return BridgeOutput(
            bridgeProtocolVersion: protocolVersion,
            ok: hasJapanese,
            text: nil,
            recognizedText: nil,
            perGlyphText: nil,
            indexableContent: nil,
            resultSource: nil,
            playbackMode: nil,
            drawingUpdateCount: nil,
            supportedLanguages: languages,
            recognizerLanguages: [],
            japaneseSupported: hasJapanese,
            recognitionVersion: PKStrokeRecognizer.recognitionVersion,
            strokeCount: nil,
            pointCount: nil,
            drawingBounds: nil,
            error: hasJapanese ? nil : BridgeError.japaneseUnavailable(languages).description
        )
    }

    @available(macOS 27.0, *)
    @MainActor
    static func recognize(_ input: RecognitionInput) async throws -> BridgeOutput {
        let (supported, languageNames, actualJapanese) = languageStatus()
        guard let actualJapanese else { throw BridgeError.japaneseUnavailable(languageNames) }
        let preferred = resolvedPreferredLanguages(
            requestedIdentifiers: input.preferredLanguages,
            supported: supported,
            actualJapanese: actualJapanese
        )
        let singleGlyphOnly = input.singleGlyphOnly ?? false
        if singleGlyphOnly {
            let glyphIndices = Set(input.strokes.compactMap { $0.glyphIndex })
            if (input.glyphCount ?? 1) > 1 || glyphIndices.contains(where: { $0 != 0 }) {
                throw BridgeError.multipleGlyphsInSingleMode
            }
        }
        let recognizer = PKStrokeRecognizer(preferredLanguages: preferred)

        let requestedMode = (input.playbackMode ?? "manual_equivalent").lowercased()
        let incremental = requestedMode == "incremental_points"
        // The native manual test panel stays alive and gives PencilKit time to
        // publish recognition.  The old command-line path queried immediately
        // after one update, so a fresh recognizer often returned nil even though
        // the same drawing worked in the panel.  Keep the application initialized,
        // poll briefly, and retry with point-by-point replay only when necessary.
        await recognizer.updateDrawing(PKDrawing())
        await Task.yield()
        var bundle = incremental
            ? try await replayDrawing(from: input, recognizer: recognizer)
            : try await makeDrawingInstantly(from: input, recognizer: recognizer)
        var recognition = await waitForRecognition(
            recognizer: recognizer, bundle: bundle, glyphCount: input.glyphCount
        )
        var usedIncrementalFallback = false
        if recognition.whole == nil && recognition.perGlyph == nil && recognition.indexable == nil && !incremental {
            await recognizer.updateDrawing(PKDrawing())
            try? await Task.sleep(nanoseconds: 120_000_000)
            bundle = try await replayDrawing(from: input, recognizer: recognizer)
            recognition = await waitForRecognition(
                recognizer: recognizer, bundle: bundle, glyphCount: input.glyphCount,
                attempts: 34, delayNanoseconds: 90_000_000
            )
            usedIncrementalFallback = true
        }
        let drawing = bundle.drawing
        let whole = recognition.whole
        let indexable = recognition.indexable
        let perGlyph = recognition.perGlyph

        let chosen: String?
        let source: String?
        if let whole {
            chosen = whole
            source = (incremental || usedIncrementalFallback) ? "incrementalRecognizedText" : "manualEquivalentRecognizedText"
        } else if let perGlyph {
            chosen = perGlyph
            source = (incremental || usedIncrementalFallback) ? "incrementalPerGlyph" : "manualEquivalentPerGlyph"
        } else if let indexable {
            chosen = indexable
            source = "indexableContent"
        } else {
            chosen = nil
            source = nil
        }

        let bounds = drawing.bounds
        let recognizerLanguageNames = await recognizer.languages.map { $0.minimalIdentifier }
        if singleGlyphOnly {
            // Explicitly clear the recognizer after this one character. The
            // Python caller starts a new bridge request for the next glyph.
            await recognizer.updateDrawing(PKDrawing())
        }
        return BridgeOutput(
            bridgeProtocolVersion: protocolVersion,
            ok: chosen != nil,
            text: chosen,
            recognizedText: whole,
            perGlyphText: perGlyph,
            indexableContent: indexable,
            resultSource: source,
            playbackMode: singleGlyphOnly
                ? ((incremental || usedIncrementalFallback) ? "single_glyph_incremental_points" : "single_glyph_manual_equivalent")
                : ((incremental || usedIncrementalFallback) ? "incremental_points" : "manual_equivalent"),
            drawingUpdateCount: bundle.updateCount,
            supportedLanguages: languageNames,
            recognizerLanguages: recognizerLanguageNames,
            japaneseSupported: true,
            recognitionVersion: PKStrokeRecognizer.recognitionVersion,
            strokeCount: drawing.strokes.count,
            pointCount: bundle.pointCount,
            drawingBounds: DrawingBounds(
                x: bounds.origin.x,
                y: bounds.origin.y,
                width: bounds.size.width,
                height: bounds.size.height
            ),
            error: chosen == nil
                ? (singleGlyphOnly
                    ? "PKStrokeRecognizer returned no text for the single submitted glyph."
                    : "PKStrokeRecognizer returned no text after drawing recognition, per-glyph subsets, or indexableContent.")
                : nil
        )
    }

    @MainActor
    static func main() async {
        let app = NSApplication.shared
        let manualPanelRequested = CommandLine.arguments.contains("--manual-panel")
        app.setActivationPolicy(manualPanelRequested ? .regular : .accessory)
        // PKStrokeRecognizer behaves reliably inside an initialized AppKit
        // process.  The manual panel already had this lifecycle; background OCR
        // did not.  finishLaunching supplies the same framework initialization
        // without showing a dock icon or window.
        app.finishLaunching()

        guard #available(macOS 27.0, *) else {
            encodeAndPrint(BridgeOutput(
                bridgeProtocolVersion: protocolVersion,
                ok: false,
                text: nil,
                recognizedText: nil,
                perGlyphText: nil,
                indexableContent: nil,
                resultSource: nil,
                playbackMode: nil,
                drawingUpdateCount: nil,
                supportedLanguages: [],
                recognizerLanguages: [],
                japaneseSupported: false,
                recognitionVersion: nil,
                strokeCount: nil,
                pointCount: nil,
                drawingBounds: nil,
                error: BridgeError.unsupportedOS.description
            ))
            return
        }

        if CommandLine.arguments.contains("--vision-candidates-batch") {
            do {
                let data = FileHandle.standardInput.readDataToEndOfFile()
                guard !data.isEmpty else { throw BridgeError.emptyInput }
                let batch = try JSONDecoder().decode(VisionCandidateBatchRequest.self, from: data)
                let results = batch.requests.map { recognizeVisionCandidates($0) }
                encodeAndPrint(VisionCandidateBatchResponse(
                    bridgeProtocolVersion: protocolVersion,
                    ok: results.contains(where: { $0.ok }),
                    results: results,
                    error: nil
                ))
            } catch {
                encodeAndPrint(VisionCandidateBatchResponse(
                    bridgeProtocolVersion: protocolVersion,
                    ok: false,
                    results: [],
                    error: String(describing: error)
                ))
            }
            return
        }

        if CommandLine.arguments.contains("--vision-candidates") {
            do {
                let data = FileHandle.standardInput.readDataToEndOfFile()
                guard !data.isEmpty else { throw BridgeError.emptyInput }
                let input = try JSONDecoder().decode(VisionCandidateRequest.self, from: data)
                encodeAndPrint(recognizeVisionCandidates(input))
            } catch {
                encodeAndPrint(VisionCandidateResponse(
                    bridgeProtocolVersion: protocolVersion,
                    ok: false,
                    correctedCandidates: [],
                    rawCandidates: [],
                    supportedLanguages: visionSupportedLanguages(),
                    error: String(describing: error)
                ))
            }
            return
        }

        if manualPanelRequested {
            let (_, languageNames, actualJapanese) = languageStatus()
            guard let actualJapanese else {
                let alert = NSAlert()
                alert.messageText = "日语手写识别不可用"
                alert.informativeText = BridgeError.japaneseUnavailable(languageNames).description
                alert.runModal()
                return
            }
            let payloadURL: URL? = {
                guard let index = CommandLine.arguments.firstIndex(of: "--payload"),
                      CommandLine.arguments.indices.contains(index + 1) else { return nil }
                return URL(fileURLWithPath: CommandLine.arguments[index + 1])
            }()
            let panel = ManualRecognitionPanel(language: actualJapanese, payloadURL: payloadURL)
            panel.show()
            app.run()
            _ = panel
            return
        }

        if CommandLine.arguments.contains("--status") {
            encodeAndPrint(statusOutput())
            return
        }

        if CommandLine.arguments.contains("--server") {
            // JSONL server mode removes repeated process/AppKit startup only.
            // Each line still enters recognize(_:) independently, creates a
            // fresh PKStrokeRecognizer, submits exactly one single-glyph
            // PKDrawing when requested, clears it, and returns one JSON line.
            while let line = readLine() {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty { continue }
                if let data = trimmed.data(using: .utf8),
                   let command = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   String(describing: command["command"] ?? "").lowercased() == "close" {
                    break
                }
                do {
                    guard let data = trimmed.data(using: .utf8) else {
                        throw BridgeError.emptyInput
                    }
                    let input = try JSONDecoder().decode(RecognitionInput.self, from: data)
                    encodeAndPrint(try await recognize(input))
                } catch {
                    let (_, languages, actualJapanese) = languageStatus()
                    encodeAndPrint(BridgeOutput(
                        bridgeProtocolVersion: protocolVersion,
                        ok: false,
                        text: nil,
                        recognizedText: nil,
                        perGlyphText: nil,
                        indexableContent: nil,
                        resultSource: nil,
                        playbackMode: nil,
                        drawingUpdateCount: nil,
                        supportedLanguages: languages,
                        recognizerLanguages: [],
                        japaneseSupported: actualJapanese != nil,
                        recognitionVersion: PKStrokeRecognizer.recognitionVersion,
                        strokeCount: nil,
                        pointCount: nil,
                        drawingBounds: nil,
                        error: String(describing: error)
                    ))
                }
            }
            return
        }

        do {
            let data = FileHandle.standardInput.readDataToEndOfFile()
            guard !data.isEmpty else { throw BridgeError.emptyInput }
            let input = try JSONDecoder().decode(RecognitionInput.self, from: data)
            encodeAndPrint(try await recognize(input))
        } catch {
            let (_, languages, actualJapanese) = languageStatus()
            encodeAndPrint(BridgeOutput(
                bridgeProtocolVersion: protocolVersion,
                ok: false,
                text: nil,
                recognizedText: nil,
                perGlyphText: nil,
                indexableContent: nil,
                resultSource: nil,
                playbackMode: nil,
                drawingUpdateCount: nil,
                supportedLanguages: languages,
                recognizerLanguages: [],
                japaneseSupported: actualJapanese != nil,
                recognitionVersion: PKStrokeRecognizer.recognitionVersion,
                strokeCount: nil,
                pointCount: nil,
                drawingBounds: nil,
                error: String(describing: error)
            ))
        }
    }
}
