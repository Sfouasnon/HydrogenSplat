import Foundation

// MARK: - hs solve --matcher

/// How `hs solve` pairs images for feature matching (`--matcher`). Exhaustive matches every
/// image with every other — n² pairs, 4–5 hours for CirclesSculpture's 844 images. Sequential
/// matches each capture with its neighbours in capture order (both eyes, the rig mate always)
/// plus a loop pass over every 8th capture. Auto is the engine's choice: sequential above 60
/// captures, else exhaustive.
public enum SolveMatcher: String, CaseIterable, Identifiable, Sendable {
    case exhaustive, sequential, auto

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .exhaustive: return "Exhaustive"
        case .sequential: return "Sequential"
        case .auto: return "Auto"
        }
    }

    /// `hs solve -p P --matcher M`. Auto is passed explicitly too, so the manifest's argv says
    /// which was asked for.
    public func solveArguments(project: String) -> [String] {
        ["solve", "-p", project, "--matcher", rawValue]
    }
}

// MARK: - hs solve --estimate

/// The cost of a solve before it runs: the single `estimate` event `hs solve -p P --estimate`
/// prints (no lock, no writes). For each matcher: images, pairs, seconds per phase (features,
/// matching, mapping, export) and the total, plus where the coefficients came from ("measured
/// on this Mac (N runs)" or "defaults").
///
/// Read tolerantly, like every event: unknown keys are ignored, and a matcher (or the whole
/// event) with nothing usable in it is simply not there — the Solve page then shows no estimate.
/// Accepted shapes, so the engine's exact layout is not a compile-time contract:
///
///     {"ev":"estimate","stage":"solve","captures":422,"auto":"sequential","calibration":"defaults",
///      "matchers":{"exhaustive":{"images":844,"pairs":355746,
///                                "seconds":{"features":..,"matching":..,"mapping":..,"export":..,"total":..}},
///                  "sequential":{…}}}
///
/// `matchers` may also be a list of entries carrying `"matcher"` (or `"name"`), or the matcher
/// names may sit at the top level. Per entry, `seconds` may be an object of phases (with or
/// without `total`) or a number (the total); `total_s`, `total` and `<phase>_s` keys are read too.
/// A missing total is the sum of the phases.
public struct SolveEstimate: Equatable, Sendable {
    public static let phases = ["features", "matching", "mapping", "export"]

    public struct Cost: Equatable, Sendable {
        public let images: Int?
        public let pairs: Int?
        /// Seconds per phase, only the ones the event gave.
        public let seconds: [String: Double]
        public let total: Double?

        public init(images: Int?, pairs: Int?, seconds: [String: Double], total: Double?) {
            self.images = images
            self.pairs = pairs
            self.seconds = seconds
            self.total = total
        }

        init?(_ v: JSONValue) {
            guard v.object != nil else { return nil }
            images = v["images"]?.int
            pairs = v["pairs"]?.int
            var phases: [String: Double] = [:]
            var total: Double? = nil
            if let obj = v["seconds"]?.object ?? v["phases"]?.object {
                for (k, x) in obj {
                    guard let s = x.double, s.isFinite, s >= 0 else { continue }
                    if k == "total" { total = s } else { phases[k] = s }
                }
            } else if let s = v["seconds"]?.double {
                total = s
            }
            for p in SolveEstimate.phases where phases[p] == nil {
                if let s = v["\(p)_s"]?.double, s.isFinite, s >= 0 { phases[p] = s }
            }
            if total == nil { total = v["total_s"]?.double ?? v["total"]?.double }
            if total == nil, !phases.isEmpty { total = phases.values.reduce(0, +) }
            if let t = total, !t.isFinite || t < 0 { total = nil }
            seconds = phases
            self.total = total
            if images == nil && pairs == nil && total == nil { return nil }
        }
    }

    /// Keyed by matcher name ("exhaustive", "sequential"; an "auto" entry if the engine sends one).
    public let costs: [String: Cost]
    public let captures: Int?
    public let calibration: String?
    /// What auto resolves to on this project, as the engine says or by its rule.
    public let autoMatcher: SolveMatcher?

    public init(costs: [String: Cost], captures: Int? = nil, calibration: String? = nil,
                autoMatcher: SolveMatcher? = nil) {
        self.costs = costs
        self.captures = captures
        self.calibration = calibration
        self.autoMatcher = autoMatcher
    }

    /// From an `estimate` event; nil for any other kind or an event with nothing usable.
    public init?(event: HSEvent) {
        guard event.kind == "estimate" else { return nil }
        self.init(json: .object(event.fields))
    }

    public init?(json j: JSONValue) {
        guard j.object != nil else { return nil }
        var costs: [String: Cost] = [:]
        let list = j["matchers"] ?? j["estimates"]
        if let d = list?.object {
            for (k, v) in d { if let c = Cost(v) { costs[k] = c } }
        } else if let a = list?.array {
            for v in a {
                if let k = v["matcher"]?.string ?? v["name"]?.string, let c = Cost(v) { costs[k] = c }
            }
        }
        if costs.isEmpty {
            for m in [SolveMatcher.exhaustive, .sequential] {
                if let v = j[m.rawValue], let c = Cost(v) { costs[m.rawValue] = c }
            }
        }
        guard !costs.isEmpty else { return nil }
        self.costs = costs
        captures = j["captures"]?.int
        calibration = j["calibration"]?.string ?? j["calibration"]?["label"]?.string

        // what "auto" means here: the engine's word first, then its rule (sequential above 60 captures)
        var said = j["auto"]?.string ?? j["auto_matcher"]?.string
        if said == nil { said = j["auto"]?["matcher"]?.string }
        if said == nil, let a = list?.object?["auto"] {
            said = a["resolves_to"]?.string ?? a["matcher"]?.string
        }
        if said == nil, let a = list?.array?.first(where: { ($0["matcher"]?.string ?? $0["name"]?.string) == "auto" }) {
            said = a["resolves_to"]?.string
        }
        if let s = said, let m = SolveMatcher(rawValue: s), m != .auto {
            autoMatcher = m
        } else if let n = captures {
            autoMatcher = n > SolveEstimate.autoThreshold ? .sequential : .exhaustive
        } else {
            autoMatcher = nil
        }
    }

    /// `hs solve`'s auto rule: sequential when the project has more captures than this.
    public static let autoThreshold = 60

    /// The cost of running with `m`. Auto is the matcher it resolves to (or the engine's own
    /// "auto" entry when it sent one and did not say which).
    public func cost(_ m: SolveMatcher) -> Cost? {
        if m == .auto {
            if let r = autoMatcher, let c = costs[r.rawValue] { return c }
            return costs["auto"]
        }
        return costs[m.rawValue]
    }

    /// "Exhaustive ≈ 4 h 40 min", "Auto (sequential) ≈ 35 min", or the bare title while nothing is known.
    public func label(_ m: SolveMatcher) -> String {
        var s = m.title
        if m == .auto, let r = autoMatcher { s += " (\(r.rawValue))" }
        if let t = cost(m)?.total { s += " " + Format.approx(t) }
        return s
    }

    /// "300 images, sequential ≈ 35 min" — the Frames page's caption beside the frame count.
    public var framesCaption: String? {
        guard let c = cost(.sequential) else { return nil }
        var parts: [String] = []
        if let n = c.images { parts.append("\(JSONValue.number(Double(n)).display) images") }
        if let t = c.total { parts.append("sequential \(Format.approx(t))") }
        return parts.isEmpty ? nil : parts.joined(separator: ", ")
    }

    /// The line under the matcher control: "844 images · 355,746 pairs exhaustive · 13,066 pairs
    /// sequential. Times measured on this Mac (3 runs)."
    public var detail: String? {
        var parts: [String] = []
        if let n = cost(.sequential)?.images ?? cost(.exhaustive)?.images {
            parts.append("\(JSONValue.number(Double(n)).display) images")
        }
        for m in [SolveMatcher.exhaustive, .sequential] {
            if let p = cost(m)?.pairs { parts.append("\(JSONValue.number(Double(p)).display) pairs \(m.rawValue)") }
        }
        let facts = parts.joined(separator: " · ")
        switch (facts.isEmpty, calibrationNote) {
        case (true, nil): return nil
        case (true, let c?): return c
        case (false, nil): return facts + "."
        case (false, let c?): return facts + ". " + c
        }
    }

    /// Where the times come from, as a sentence: the engine's label ("measured on this Mac
    /// (3 runs)") after "Times", or a plain word on the seeded defaults.
    public var calibrationNote: String? {
        guard let c = calibration?.trimmingCharacters(in: .whitespaces), !c.isEmpty else { return nil }
        if c == "defaults" { return "Times are the engine's defaults until a solve has been timed on this Mac." }
        return "Times \(c)."
    }

    public static func arguments(project: String) -> [String] {
        ["solve", "-p", project, "--estimate"]
    }

    /// Runs `hs solve -p P --estimate` and returns the first `estimate` event it prints, or nil
    /// (an engine without `--estimate`, a project with nothing to solve, a crash — the page then
    /// shows no estimate). Cheap: the engine takes no lock and writes nothing.
    public static func load(config: EngineConfig, project: String) async -> SolveEstimate? {
        await withCheckedContinuation { (cont: CheckedContinuation<SolveEstimate?, Never>) in
            let r = ToolRunner(executable: config.hsPath, arguments: arguments(project: project),
                               environment: config.environment(), workingDirectory: config.repoRoot)
            var found: SolveEstimate?
            do {
                try r.start(onEvents: { evs in
                    for e in evs where found == nil { found = SolveEstimate(event: e) }
                }, onExit: { _ in
                    _ = r      // the runner lives until the child has exited
                    cont.resume(returning: found)
                })
            } catch {
                cont.resume(returning: nil)
            }
        }
    }
}

extension Format {
    /// A duration that has not happened yet, as a rough figure: "≈ 35 min", "≈ 4 h 40 min".
    /// Minutes under an hour; past an hour, to the nearest 5 minutes — an estimate that says
    /// 4 h 37 min claims a precision it does not have.
    public static func approx(_ seconds: Double?) -> String {
        guard let s = seconds, s.isFinite, s >= 0 else { return "—" }
        if s < 30 { return "< 1 min" }
        let m = Int((s / 60).rounded())
        if m < 60 { return "≈ \(max(m, 1)) min" }
        let m5 = Int((Double(m) / 5).rounded()) * 5
        let h = m5 / 60, r = m5 % 60
        return r == 0 ? "≈ \(h) h" : "≈ \(h) h \(r) min"
    }
}
