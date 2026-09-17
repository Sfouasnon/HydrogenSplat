import Foundation
import Combine

public struct ProjectSummary: Identifiable, Sendable {
    public var id: String { path }
    public let path: String
    public let folderName: String
    public let manifest: Manifest?
    public let manifestError: String?
    public let lock: LockInfo?

    public struct LockInfo: Sendable, Hashable {
        public let pid: Int?
        public let stage: String?
        public let alive: Bool
    }

    public var displayName: String { folderName }
    public var created: Date? { manifest?.created }
}

/// The folder of projects. Each subfolder with a manifest.json is a project; folders whose
/// name starts with "_" (e.g. _clipqa) are tools' scratch space and are skipped.
@MainActor
public final class ProjectStore: ObservableObject {
    @Published public private(set) var projects: [ProjectSummary] = []
    @Published public private(set) var lastError: String?
    @Published public var root: String {
        didSet { if root != oldValue { reload() } }
    }

    private var lastStamp = ""

    public init(root: String) {
        self.root = root
        reload()
        lastStamp = stamp()
    }

    public func reload() {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: root) else {
            projects = []
            lastError = "cannot read \(root)"
            return
        }
        lastError = nil
        var out: [ProjectSummary] = []
        for n in names where !n.hasPrefix("_") && !n.hasPrefix(".") {
            let dir = (root as NSString).appendingPathComponent(n)
            let mpath = (dir as NSString).appendingPathComponent("manifest.json")
            guard fm.fileExists(atPath: mpath) else { continue }
            out.append(ProjectStore.load(dir))
        }
        out.sort { a, b in
            switch (a.created, b.created) {
            case let (x?, y?) where x != y: return x > y
            default: return a.folderName > b.folderName
            }
        }
        projects = out
        lastStamp = stamp()
    }

    public func project(at path: String) -> ProjectSummary? {
        projects.first { $0.path == path }
    }

    /// Reload only when something on disk actually changed. A stage run from Terminal writes
    /// manifest.json and removes .hs.lock with nothing to tell the app, so without this the
    /// sidebar and the stages table keep showing whatever was true at the last reload -- a
    /// finished run still reading "train running", with no progress bar, because the lock it
    /// would follow is already gone. Stamping mtime+size of those two files per project is a
    /// handful of stats; reload() itself re-reads every manifest, so it stays behind the stamp.
    public func refreshIfChanged() {
        let now = stamp()
        if now != lastStamp {
            lastStamp = now
            reload()
        }
    }

    private func stamp() -> String {
        let fm = FileManager.default
        var parts: [String] = []
        for n in ((try? fm.contentsOfDirectory(atPath: root)) ?? []).sorted()
        where !n.hasPrefix("_") && !n.hasPrefix(".") {
            let dir = (root as NSString).appendingPathComponent(n)
            for f in ["manifest.json", ".hs.lock"] {
                let p = (dir as NSString).appendingPathComponent(f)
                let a = try? fm.attributesOfItem(atPath: p)
                let t = (a?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
                let s = (a?[.size] as? Int) ?? -1
                parts.append("\(n)/\(f):\(t):\(s)")
            }
        }
        return parts.joined(separator: "|")
    }

    public nonisolated static func load(_ dir: String) -> ProjectSummary {
        let mpath = (dir as NSString).appendingPathComponent("manifest.json")
        var manifest: Manifest?
        var err: String?
        if let data = FileManager.default.contents(atPath: mpath) {
            manifest = Manifest(data: data)
            if manifest == nil { err = "manifest.json is not valid JSON" }
        } else {
            err = "manifest.json unreadable"
        }
        return ProjectSummary(path: dir, folderName: (dir as NSString).lastPathComponent,
                              manifest: manifest, manifestError: err, lock: readLock(dir))
    }

    /// .hs.lock: {"pid": N, "stage": "...", ...}
    public nonisolated static func readLock(_ dir: String) -> ProjectSummary.LockInfo? {
        let p = (dir as NSString).appendingPathComponent(".hs.lock")
        guard let data = FileManager.default.contents(atPath: p) else { return nil }
        let v = JSONValue.parse(data)
        let pid = v?["pid"]?.int
        let alive = pid.map { kill(pid_t($0), 0) == 0 || errno == EPERM } ?? false
        return .init(pid: pid, stage: v?["stage"]?.string, alive: alive)
    }

    /// Basenames of every ingested clip, to mark phone clips that already have a project.
    public var ingestedClips: [String: String] {
        var d: [String: String] = [:]
        for p in projects {
            if let c = p.manifest?.clipName { d[c] = p.folderName }
        }
        return d
    }

    /// "VID_20260915_145235_2x1.h4v" → "2026-09-15_145235"; the label replaces the time.
    public nonisolated static func suggestedName(clip: String, label: String = "") -> String {
        let base = (clip as NSString).lastPathComponent
        let clean = label.trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: "[^A-Za-z0-9_-]+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        if let m = base.range(of: #"^VID_(\d{4})(\d{2})(\d{2})_(\d{6})"#, options: .regularExpression) {
            let s = String(base[m])
            let d = s.dropFirst(4)
            let date = "\(d.prefix(4))-\(d.dropFirst(4).prefix(2))-\(d.dropFirst(6).prefix(2))"
            let time = String(d.suffix(6))
            return "\(date)_\(clean.isEmpty ? time : clean)"
        }
        let stem = ((base as NSString).deletingPathExtension)
        return clean.isEmpty ? stem : clean
    }

    /// `name`, or `name-2`, `name-3`… so an ingest never lands in an existing project.
    public func uniqueFolder(_ name: String) -> String {
        let fm = FileManager.default
        var candidate = name
        var i = 2
        while fm.fileExists(atPath: (root as NSString).appendingPathComponent(candidate)) {
            candidate = "\(name)-\(i)"
            i += 1
        }
        return (root as NSString).appendingPathComponent(candidate)
    }
}
