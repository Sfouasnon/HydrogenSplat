import Foundation

/// A project's manifest.json (engine/hs/project.py), read tolerantly.
public struct CheckResult: Hashable, Sendable, Identifiable {
    public var id: String { name + "|" + (value?.display ?? "") }
    public let name: String
    public let ok: Bool
    public let value: JSONValue?
    public let needsHuman: Bool

    public init(name: String, ok: Bool, value: JSONValue?, needsHuman: Bool = false) {
        self.name = name
        self.ok = ok
        self.value = value
        self.needsHuman = needsHuman
    }

    init?(_ v: JSONValue) {
        guard let name = v["name"]?.string else { return nil }
        self.init(name: name, ok: v["ok"]?.bool ?? false, value: v["value"],
                  needsHuman: v["needs_human"]?.bool ?? false)
    }
}

public enum StageStatus: String, Sendable, CaseIterable {
    case pending, running, done, failed, stale, unknown

    public init(raw: String?) {
        self = StageStatus(rawValue: raw ?? "pending") ?? .unknown
    }
}

public struct StageState: Hashable, Sendable, Identifiable {
    public var id: String { name }
    public let name: String
    public let status: StageStatus
    public let started: Date?
    public let finished: Date?
    public let argv: [String]
    public let metrics: [String: JSONValue]
    public let checks: [CheckResult]
    public let artifacts: [String]
    public let error: String?
    public let pid: Int?

    init(name: String, _ v: JSONValue) {
        self.name = name
        status = StageStatus(raw: v["status"]?.string)
        started = Manifest.date(v["started"]?.string)
        finished = Manifest.date(v["finished"]?.string)
        argv = v["argv"]?.array?.compactMap { $0.string } ?? []
        metrics = v["metrics"]?.object ?? [:]
        checks = v["checks"]?.array?.compactMap(CheckResult.init) ?? []
        artifacts = v["artifacts"]?.array?.compactMap { $0["path"]?.string } ?? []
        error = v["error"]?.string
        pid = v["pid"]?.int
    }

    public var duration: TimeInterval? {
        guard let s = started, let f = finished else { return nil }
        return f.timeIntervalSince(s)
    }

    public var failedChecks: Int { checks.filter { !$0.ok }.count }

    /// Metrics that are worth a table row: long lists (growth curves) are summarised.
    public var displayMetrics: [(String, String)] {
        metrics.keys.sorted().map { k in
            let v = metrics[k]!
            if let a = v.array, a.count > 12 { return (k, "\(a.count) entries") }
            return (k, v.display)
        }
    }
}

public struct Manifest: Sendable {
    /// The engine's chain (project.STAGES) with the two dataset operations placed where they run.
    public static let stageOrder = ["ingest", "select", "solve", "exposure", "masks", "train",
                                    "archive", "move", "prune", "render", "views"]

    public let raw: JSONValue
    public let name: String
    public let created: Date?
    public let profileID: String?
    public let clipPath: String?
    public let clipMD5: String?
    public let originalPath: String?
    public let probe: [String: JSONValue]
    public let stages: [StageState]

    public init?(data: Data) {
        guard let v = JSONValue.parse(data), v.object != nil else { return nil }
        raw = v
        name = v["name"]?.string ?? "?"
        created = Manifest.date(v["created"]?.string)
        profileID = v["profile_id"]?.string
        clipPath = v["source"]?["clip"]?.string
        clipMD5 = v["source"]?["md5"]?.string
        originalPath = v["source"]?["original_path"]?.string
        probe = v["source"]?["probe"]?.object ?? [:]
        let st = v["stages"]?.object ?? [:]
        let known = Manifest.stageOrder.filter { st[$0] != nil }
        let extra = st.keys.filter { !Manifest.stageOrder.contains($0) }.sorted()
        stages = (known + extra).map { StageState(name: $0, st[$0]!) }
    }

    public func stage(_ name: String) -> StageState? { stages.first { $0.name == name } }

    public var clipName: String? { clipPath.map { ($0 as NSString).lastPathComponent } }

    /// The furthest stage that is done, for the project list.
    public var lastDone: String? {
        stages.last(where: { $0.status == .done && $0.name != "archive" })?.name
    }

    static let isoFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZ"   // now_iso(): %Y-%m-%dT%H:%M:%S%z
        return f
    }()

    static func date(_ s: String?) -> Date? {
        guard let s = s else { return nil }
        return isoFormatter.date(from: s)
    }
}
