import Foundation
import AppKit

@MainActor
final class AppModel: ObservableObject {
    static weak var current: AppModel?

    @Published private(set) var stage: Stage = .checking
    @Published private(set) var statusLine: String = "Checking runtime…"
    @Published private(set) var log: String = ""
    @Published var lastError: String?

    private let paths = RuntimePaths()
    private let runtime = PythonRuntime()
    private var child: Process?
    private var childPipe: Pipe?

    enum Stage {
        case checking, preparing, ready, running, error
    }

    init() {
        AppModel.current = self
    }

    // MARK: - Lifecycle

    /// Runs once at launch: prepares the standalone Python and the writable
    /// source copy, then reports "ready". The GUI itself starts only when the
    /// user clicks "Open workspace" (or on first successful prepare).
    func startupCheck() {
        guard stage == .checking else { return }
        Task {
            stage = .preparing
            do {
                try await runtime.prepare { [weak self] line in
                    Task { @MainActor in self?.statusLine = line }
                }
                stage = .ready
                statusLine = "Runtime ready — Python \(RuntimePaths.pythonVersion), no OCR models bundled."
                appendLog("BOOT_STAGE=ready")
            } catch {
                stage = .error
                lastError = error.localizedDescription
                statusLine = "Preparation failed."
                appendLog("ERROR \(error.localizedDescription)")
            }
        }
    }

    /// Starts the Python GUI through bootstrap.py so dependency installation
    /// stays identical to launcher.zsh.
    func openWorkspace() {
        guard child == nil, stage == .ready else { return }
        appendLog("BOOT_STAGE=prepare_application_dependencies")
        stage = .running
        statusLine = "Starting Python GUI…"

        var environment = Self.cleanedEnvironment()
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        environment["QT_API"] = "pyside6"
        environment["QT_ENABLE_HIGHDPI_SCALING"] = "1"
        environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        environment["PIP_NO_INPUT"] = "1"

        let process = Process()
        process.executableURL = paths.pythonBinary
        process.arguments = [
            "-X", "faulthandler",
            "bootstrap.py", "--install-main-deps", "--launch",
        ]
        process.currentDirectoryURL = paths.pythonWorkingDir
        process.environment = environment

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        childPipe = pipe

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.appendLog(text) }
        }

        process.terminationHandler = { [weak self] proc in
            Task { @MainActor in
                guard let self else { return }
                self.child = nil
                self.childPipe?.fileHandleForReading.readabilityHandler = nil
                let status = proc.terminationStatus
                if status == 0 {
                    self.statusLine = "Python GUI closed."
                    self.stage = .ready
                } else {
                    self.statusLine = "Python GUI exited with code \(status)."
                    self.stage = .ready
                }
                self.appendLog("BOOT_STAGE=exit status=\(status)")
            }
        }

        do {
            try process.run()
            child = process
            statusLine = "Python GUI is running."
        } catch {
            stage = .ready
            statusLine = "Failed to start the Python GUI."
            lastError = error.localizedDescription
            appendLog("ERROR launch \(error.localizedDescription)")
        }
    }

    func stopChild() {
        childPipe?.fileHandleForReading.readabilityHandler = nil
        child?.terminate()
        child = nil
    }

    func openLogsInFinder() {
        NSWorkspace.shared.activateFileViewerSelecting([paths.logFile])
    }

    func revealSupportFolder() {
        NSWorkspace.shared.activateFileViewerSelecting([paths.supportDir])
    }

    // MARK: - Helpers

    /// Mirrors clean_injected_environment() from launcher.zsh so a shell
    /// exported PYTHONPATH or DYLD_* cannot break the bundled runtime.
    static func cleanedEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        for key in [
            "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE",
            "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH",
            "QML2_IMPORT_PATH", "QML_IMPORT_PATH",
            "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH",
            "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_FALLBACK_FRAMEWORK_PATH",
        ] {
            env.removeValue(forKey: key)
        }
        return env
    }

    private func appendLog(_ text: String) {
        log += text.hasSuffix("\n") ? text : text + "\n"
        if log.count > 200_000 {
            log = String(log.suffix(150_000))
        }
    }
}
