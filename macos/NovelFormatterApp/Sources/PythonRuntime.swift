import Foundation
import CryptoKit

/// Prepares the standalone CPython runtime and the writable app-source copy.
/// Mirrors launcher.zsh: downloads come from the same mirrors with SHA-256
/// verification, and the OCR policy is unchanged (models install only after
/// an explicit OCR-start confirmation inside the Python GUI).
enum RuntimeError: LocalizedError {
    case downloadFailed(String)
    case checksumMismatch
    case archiveInvalid
    case archMismatch(String)

    var errorDescription: String? {
        switch self {
        case .downloadFailed(let target):
            return "Download failed: \(target). Check the network and try again."
        case .checksumMismatch:
            return "SHA-256 verification failed for the standalone Python archive. The file was deleted."
        case .archiveInvalid:
            return "The standalone Python archive is corrupted and was deleted."
        case .archMismatch(let detail):
            return "Standalone Python architecture mismatch, runtime was cleared: \(detail)"
        }
    }
}

actor PythonRuntime {
    private let paths = RuntimePaths()
    private let fm = FileManager.default

    /// Ensures a working standalone Python, then copies the bundled source
    /// into Application Support when the app runs standalone (no repository
    /// next to the bundle).
    func prepare(progress: @Sendable (String) -> Void) async throws {
        try fm.createDirectory(at: paths.supportDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: paths.downloadsDir, withIntermediateDirectories: true)

        if !fm.isExecutableFile(atPath: paths.pythonBinary.path) {
            try await installPython(progress: progress)
        }
        try await verifyArchitecture()
        try syncEmbeddedSourceIfNeeded(progress: progress)
    }

    // MARK: - Source copy

    private func syncEmbeddedSourceIfNeeded(progress: @Sendable (String) -> Void) {
        guard let bundled = paths.bundledSource,
              fm.fileExists(atPath: bundled.appendingPathComponent("gui_pyside6.py").path) else {
            return
        }
        progress("Copying application files…")
        let destination = paths.appSourceDir
        try? fm.createDirectory(at: destination, withIntermediateDirectories: true)
        let stateFile = destination
            .appendingPathComponent(".runtime/source_update_state.json")
        let preserved = fm.fileExists(atPath: stateFile.path)
            && versionIsAtLeast(readVersion(at: destination), readVersion(at: bundled))
        if preserved {
            log("BOOT_STAGE=source_standalone_git_preserved")
            return
        }
        syncDirectory(source: bundled, destination: destination)
        log("BOOT_STAGE=source_standalone")
    }

    private func readVersion(at root: URL) -> String {
        (try? String(contentsOf: root.appendingPathComponent("VERSION"),
                     encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    private func versionIsAtLeast(_ a: String, _ b: String) -> Bool {
        guard !a.isEmpty, !b.isEmpty else { return false }
        let pa = a.replacingOccurrences(of: "v", with: "")
            .split(separator: ".").compactMap { Int($0) }
        let pb = b.replacingOccurrences(of: "v", with: "")
            .split(separator: ".").compactMap { Int($0) }
        for i in 0..<max(pa.count, pb.count) {
            let x = i < pa.count ? pa[i] : 0
            let y = i < pb.count ? pb[i] : 0
            if x != y { return x > y }
        }
        return true
    }

    private static let excludedNames: Set<String> = [
        ".git", ".runtime", ".model-cache", ".ocr-runtimes",
        ".manual-model-updates", "__pycache__", ".pytest_cache",
    ]

    private func syncDirectory(source: URL, destination: URL) {
        guard let enumerator = fm.enumerator(
            at: source,
            includingPropertiesForKeys: [.isDirectoryKey]
        ) else { return }
        for case let item as URL in enumerator {
            let rel = item.path.replacingOccurrences(of: source.path + "/", with: "")
            let name = item.lastPathComponent
            let isDir = ((try? item.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory) ?? false
            if Self.excludedNames.contains(name) || name.hasSuffix(".pyc") || name.hasSuffix(".log") {
                if isDir { enumerator.skipDescendants() }
                continue
            }
            let target = destination.appendingPathComponent(rel)
            if isDir {
                try? fm.createDirectory(at: target, withIntermediateDirectories: true)
            } else {
                try? fm.removeItem(at: target)
                try? fm.copyItem(at: item, to: target)
            }
        }
    }

    // MARK: - Python download

    private func installPython(progress: @Sendable (String) -> Void) async throws {
        let archivePath = paths.downloadsDir.appendingPathComponent(paths.pythonArchiveName)
        let sumsPath = paths.downloadsDir.appendingPathComponent("SHA256SUMS-\(RuntimePaths.pythonRelease)")

        if !fm.fileExists(atPath: archivePath.path) || fm.contents(atPath: archivePath.path)?.isEmpty != false {
            progress("Downloading standalone Python \(RuntimePaths.pythonVersion)…")
            try await download(to: archivePath, candidates: paths.pythonArchiveURLs())
        }
        if !fm.fileExists(atPath: sumsPath.path) {
            progress("Downloading checksum file…")
            try await download(to: sumsPath, candidates: paths.sumsURLs())
        }

        progress("Verifying SHA-256…")
        guard let sums = try? String(contentsOf: sumsPath, encoding: .utf8),
              let expected = expectedChecksum(from: sums, filename: paths.pythonArchiveName),
              expected == sha256(of: archivePath) else {
            try? fm.removeItem(at: archivePath)
            try? fm.removeItem(at: sumsPath)
            throw RuntimeError.checksumMismatch
        }

        progress("Extracting standalone Python…")
        let staging = paths.supportDir.appendingPathComponent("native-runtime-v1.new", isDirectory: true)
        try? fm.removeItem(at: staging)
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        let ok = await untar(archive: archivePath, into: staging)
        guard ok, fm.isExecutableFile(atPath: staging.appendingPathComponent("python/bin/python3").path) else {
            try? fm.removeItem(at: archivePath)
            try? fm.removeItem(at: staging)
            throw RuntimeError.archiveInvalid
        }
        try? fm.removeItem(at: paths.runtimeDir)
        try fm.moveItem(at: staging, to: paths.runtimeDir)
        log("BOOT_STAGE=install_standalone_python_done")
    }

    private func verifyArchitecture() async throws {
        let output = try await runProcess(
            executable: paths.pythonBinary,
            arguments: ["-c", "import platform; print(platform.machine())"]
        )
        let machine = output.trimmingCharacters(in: .whitespacesAndNewlines)
        #if arch(arm64)
        let expected = "arm64"
        #else
        let expected = "x86_64"
        #endif
        if machine != expected {
            try? fm.removeItem(at: paths.runtimeDir)
            throw RuntimeError.archMismatch(machine)
        }
    }

    // MARK: - Helpers

    private func download(to destination: URL, candidates: [URL]) async throws {
        for url in candidates {
            do {
                let (data, response) = try await URLSession.shared.data(from: url)
                if let http = response as? HTTPURLResponse, http.statusCode == 200, !data.isEmpty {
                    try data.write(to: destination, options: .atomic)
                    return
                }
            } catch {
                log("download attempt failed: \(url.absoluteString): \(error.localizedDescription)")
            }
        }
        throw RuntimeError.downloadFailed(candidates.first?.absoluteString ?? "?")
    }

    private func expectedChecksum(from sums: String, filename: String) -> String? {
        for line in sums.split(separator: "\n") {
            let parts = line.split(separator: " ", omittingEmptySubsequences: true)
            if parts.count == 2, parts[1].trimmingCharacters(in: CharacterSet(charactersIn: "*")) == filename {
                return String(parts[0])
            }
        }
        return nil
    }

    private func sha256(of url: URL) -> String {
        guard let data = try? Data(contentsOf: url) else { return "" }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func untar(archive: URL, into directory: URL) async -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xzf", archive.path, "-C", directory.path]
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }

    private func runProcess(executable: URL, arguments: [String]) async throws -> String {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = executable
        process.arguments = arguments
        process.standardOutput = pipe
        try process.run()
        process.waitUntilExit()
        return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    }

    private func log(_ message: String) {
        try? fm.createDirectory(at: paths.logDir, withIntermediateDirectories: true)
        let line = "\(Date()): \(message)\n"
        if fm.fileExists(atPath: paths.logFile.path),
           let handle = try? FileHandle(forWritingTo: paths.logFile) {
            handle.seekToEndOfFile()
            handle.write(Data(line.utf8))
            try? handle.close()
        } else {
            try? line.write(to: paths.logFile, atomically: true, encoding: .utf8)
        }
    }
}
