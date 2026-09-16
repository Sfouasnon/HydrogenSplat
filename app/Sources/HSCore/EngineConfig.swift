import Foundation

/// Where the engine lives and how to launch it. A GUI app inherits a minimal PATH
/// (/usr/bin:/bin:/usr/sbin:/sbin), so Homebrew's ffmpeg, ffprobe and adb would be invisible
/// to every child without `extraPath`.
public struct EngineConfig: Codable, Equatable, Sendable {
    public var repoRoot: String
    public var projectsRoot: String
    public var hsPath: String
    public var extraPath: String

    public init(repoRoot: String, projectsRoot: String? = nil, hsPath: String? = nil,
                extraPath: String = "/opt/homebrew/bin:/usr/local/bin") {
        let root = (repoRoot as NSString).expandingTildeInPath
        self.repoRoot = root
        self.projectsRoot = projectsRoot ?? (root as NSString).appendingPathComponent("Projects")
        self.hsPath = hsPath ?? (root as NSString).appendingPathComponent(".venv/bin/hs")
        self.extraPath = extraPath
    }

    /// The repo this source file was built from (app/Sources/HSCore/ → up four), else the
    /// usual checkout location.
    public static func guess(sourceFile: String = #filePath) -> EngineConfig {
        var url = URL(fileURLWithPath: sourceFile)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        let fm = FileManager.default
        if fm.fileExists(atPath: url.appendingPathComponent("engine/hs/cli.py").path) {
            return EngineConfig(repoRoot: url.path)
        }
        return EngineConfig(repoRoot: "~/Desktop/Apps/HydrogenSplat")
    }

    public var eventsDir: String { (repoRoot as NSString).appendingPathComponent("engine/tests/events") }

    public func environment(base: [String: String] = ProcessInfo.processInfo.environment) -> [String: String] {
        var env = base
        let current = env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        let venvBin = (hsPath as NSString).deletingLastPathComponent
        var parts: [String] = []
        for p in [venvBin] + extraPath.split(separator: ":").map(String.init) + current.split(separator: ":").map(String.init)
        where !p.isEmpty && !parts.contains(p) {
            parts.append(p)
        }
        env["PATH"] = parts.joined(separator: ":")
        env["PYTHONUNBUFFERED"] = "1"
        return env
    }

    public enum Problem: Equatable, Sendable {
        case repoMissing(String)
        case hsMissing(String)
    }

    public var problems: [Problem] {
        let fm = FileManager.default
        var out: [Problem] = []
        if !fm.fileExists(atPath: (repoRoot as NSString).appendingPathComponent("engine/hs/cli.py")) {
            out.append(.repoMissing(repoRoot))
        }
        if !fm.isExecutableFile(atPath: hsPath) {
            out.append(.hsMissing(hsPath))
        }
        return out
    }
}
