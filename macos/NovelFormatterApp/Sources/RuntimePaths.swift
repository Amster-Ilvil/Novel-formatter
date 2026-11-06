import Foundation

/// Filesystem layout shared with packaging/launcher.zsh so the Swift shell
/// and the classic zsh shell reuse one standalone Python runtime, one
/// downloads cache and one writable app-source tree.
struct RuntimePaths {
    static let pythonRelease = "20251010"
    static let pythonVersion = "3.12.12"

    let fileManager = FileManager.default

    var supportDir: URL {
        fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Novel Formatter Studio", isDirectory: true)
    }

    var runtimeDir: URL {
        supportDir.appendingPathComponent("native-runtime-v1", isDirectory: true)
    }

    var downloadsDir: URL {
        supportDir.appendingPathComponent("downloads", isDirectory: true)
    }

    var appSourceDir: URL {
        supportDir.appendingPathComponent("app-source", isDirectory: true)
    }

    var logDir: URL {
        supportDir.appendingPathComponent("logs", isDirectory: true)
    }

    var logFile: URL {
        logDir.appendingPathComponent("swift-shell.log")
    }

    var pythonBinary: URL {
        runtimeDir.appendingPathComponent("python/bin/python3")
    }

    /// Working directory for the Python GUI. The bundled copy is used only
    /// after it has been synced into Application Support, so it is always
    /// the same writable tree.
    var pythonWorkingDir: URL {
        appSourceDir
    }

    /// Python core embedded into the signed bundle by
    /// packaging/build_macos_swift_app.sh. It is copied into appSourceDir on
    /// first run so model/runtime directories stay writable.
    var bundledSource: URL? {
        Bundle.main.resourceURL?.appendingPathComponent("app", isDirectory: true)
    }

    var pythonArchiveName: String {
        let arch = Self.pythonArch
        return "cpython-\(Self.pythonVersion)+\(Self.pythonRelease)-\(arch)-apple-darwin-install_only_stripped.tar.gz"
    }

    static var pythonArch: String {
        #if arch(arm64)
        return "aarch64"
        #else
        return "x86_64"
        #endif
    }

    func pythonArchiveURLs() -> [URL] {
        let name = pythonArchiveName
        let release = Self.pythonRelease
        return [
            URL(string: "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/\(release)/\(name)"),
            URL(string: "https://github.com/astral-sh/python-build-standalone/releases/download/\(release)/\(name)"),
        ].compactMap { $0 }
    }

    func sumsURLs() -> [URL] {
        let release = Self.pythonRelease
        return [
            URL(string: "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/\(release)/SHA256SUMS"),
            URL(string: "https://github.com/astral-sh/python-build-standalone/releases/download/\(release)/SHA256SUMS"),
        ].compactMap { $0 }
    }
}
