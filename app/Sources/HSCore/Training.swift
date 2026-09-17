import Foundation
import Combine
import IOKit.ps

// MARK: - Settings → hs arguments

/// What the Train panel exposes, and the `hs` commands it turns into.
/// Defaults are the rig6 schedule `hs train` has always used.
/// The captures a project holds, in rig order: `cap000…` pairs from a Hydrogen clip, or one
/// name per camera (`GA`, `GB`, …) from an array. Read from solve/coverage.json, which the
/// solve writes in the same frame and order the path builders and `hs views` use.
public struct CaptureSet: Equatable, Sendable {
    public var names: [String]
    public var stereo: Bool

    public init(names: [String], stereo: Bool) {
        self.names = names
        self.stereo = stereo
    }

    /// A stereo set of `count` captures named the way `hs solve` names them.
    public init(count: Int) {
        self.names = (0..<max(count, 0)).map { String(format: "cap%03d", $0) }
        self.stereo = true
    }

    public var count: Int { names.count }
    /// Views Brush trains on: two per capture on a stereo rig, one per camera on an array.
    public var views: Int { stereo ? count * 2 : count }
    public var noun: String { stereo ? "capture" : "camera" }

    public static func read(project: String) -> CaptureSet {
        let p = (project as NSString).appendingPathComponent("solve/coverage.json")
        guard let d = FileManager.default.contents(atPath: p), let j = JSONValue.parse(d) else {
            return CaptureSet(names: [], stereo: true)
        }
        // "stereo" is absent from every coverage.json written before arrays existed: those are pairs
        let stereo = j["stereo"]?.bool ?? true
        let names = (j["captures"]?.array ?? []).compactMap { $0["name"]?.string }
        if names.isEmpty {
            return CaptureSet(count: j["n_captures"]?.int ?? 0)
        }
        return CaptureSet(names: names, stereo: stereo)
    }
}

public struct TrainSettings: Equatable, Sendable {
    public var totalIters = 40_000
    public var growthStopIter = 30_000
    public var refineEvery = 130
    /// nil = Brush's default (0.5 of the image)
    public var splitAtScreenSize: Double? = nil
    /// Brush ≥ #541: 0.1 default, 0 = off
    public var minScaleFactor = 0.1
    /// nil = Brush's default (0.1)
    public var backgroundNoise: Double? = nil
    /// 0 = off; otherwise hold out every Nth capture (both eyes) starting at `holdoutStart`
    public var holdoutEvery = 10
    public var holdoutStart = 5
    public var useMasks = true
    /// extra views to leave out, e.g. "L/cap064,R/cap069"
    public var excludeExtra = ""
    public var extraBrushArgs = ""
    public var archiveName = ""
    public var scoreViews = true
    public var scoreBothEyes = false
    public var subjectMM: Double? = nil

    public init() {}

    public func holdoutCaptures(total: Int) -> [Int] {
        guard holdoutEvery > 0, total > 0 else { return [] }
        return Array(stride(from: holdoutStart, to: total, by: holdoutEvery))
    }

    public func excludedViews(in set: CaptureSet) -> [String] {
        let eyes = set.stereo ? ["L", "R"] : ["L"]      // an array's single view per camera lives in images/L
        var v = holdoutCaptures(total: set.count).flatMap { c in eyes.map { "\($0)/" + set.names[c] } }
        for x in excludeExtra.split(whereSeparator: { $0 == "," || $0 == " " }) where !x.isEmpty {
            let s = String(x)
            if !v.contains(s) { v.append(s) }
        }
        return v
    }

    public var brushArgs: String {
        var parts: [String] = []
        if let n = backgroundNoise { parts.append("--background-noise-strength \(TrainSettings.num(n))") }
        let extra = extraBrushArgs.trimmingCharacters(in: .whitespaces)
        if !extra.isEmpty { parts.append(extra) }
        return parts.joined(separator: " ")
    }

    public func trainArguments(project: String, captures: Int) -> [String] {
        trainArguments(project: project, in: CaptureSet(count: captures))
    }

    public func steps(project: String, captures: Int) -> [RunQueue.Step] {
        steps(project: project, in: CaptureSet(count: captures))
    }

    public func trainArguments(project: String, in set: CaptureSet) -> [String] {
        var a = ["train", "-p", project,
                 "--total-train-iters", String(totalIters),
                 "--growth-stop-iter", String(growthStopIter),
                 "--refine-every", String(refineEvery),
                 "--min-scale-factor=\(TrainSettings.num(minScaleFactor))"]
        if let s = splitAtScreenSize { a.append("--split-at-screen-size=\(TrainSettings.num(s))") }
        let ex = excludedViews(in: set)
        if !ex.isEmpty { a.append("--exclude=\(ex.joined(separator: ","))") }
        if !useMasks { a.append("--no-masks") }
        let b = brushArgs
        if !b.isEmpty { a.append("--brush-args=\(b)") }
        return a
    }

    public var archiveNameProblem: String? {
        let n = archiveName.trimmingCharacters(in: .whitespaces)
        if n.isEmpty { return nil }
        if n.range(of: #"^[A-Za-z0-9][A-Za-z0-9._-]*$"#, options: .regularExpression) == nil {
            return "letters, digits, . _ - only"
        }
        return nil
    }

    /// train → archive (if named) → views (if asked). Each step only runs if the one before succeeded.
    public func steps(project: String, in set: CaptureSet) -> [RunQueue.Step] {
        var s = [RunQueue.Step(title: "Train", arguments: trainArguments(project: project, in: set))]
        let name = archiveName.trimmingCharacters(in: .whitespaces)
        if !name.isEmpty {
            s.append(.init(title: "Archive \(name)", arguments: ["archive", "-p", project, "--name", name]))
        }
        if scoreViews {
            let held = holdoutCaptures(total: set.count)
            // never the bare "views" name: that is the in-sample report hs views writes by default
            let label = name.isEmpty ? (held.isEmpty ? "views_latest" : "holdout_latest") : "views_\(name)"
            var base = ["views", "-p", project]
            if !held.isEmpty { base += ["--captures", held.map(String.init).joined(separator: ",")] }
            if let mm = subjectMM { base += ["--subject-mm", TrainSettings.num(mm)] }
            s.append(.init(title: "Score views (L)", arguments: base + ["--name", label]))
            if scoreBothEyes && set.stereo {       // an array has no second eye to subtract
                s.append(.init(title: "Score views (R)", arguments: base + ["--eye", "R", "--name", label + "_R"]))
            }
        }
        return s
    }

    static func num(_ x: Double) -> String {
        x == x.rounded() && abs(x) < 1e9 ? String(Int(x)) : String(format: "%g", x)
    }
}

// MARK: - A queue of hs runs

@MainActor
public final class RunQueue: ObservableObject {
    public struct Step: Hashable, Sendable {
        public let title: String
        public let arguments: [String]
        public init(title: String, arguments: [String]) {
            self.title = title
            self.arguments = arguments
        }
    }

    public enum State: Equatable { case idle, running, finished, failed, cancelled }

    public let steps: [Step]
    @Published public private(set) var index = 0
    @Published public private(set) var sessions: [RunSession] = []
    @Published public private(set) var state: State = .idle
    public var onFinish: ((RunQueue) -> Void)?
    /// Called as each step's session is created, before it starts.
    public var onStep: ((RunSession) -> Void)?
    private let config: EngineConfig

    public init(config: EngineConfig, steps: [Step]) {
        self.config = config
        self.steps = steps
    }

    public var current: RunSession? { sessions.last }
    public var isRunning: Bool { state == .running }

    public func start() {
        guard state == .idle, !steps.isEmpty else { return }
        state = .running
        run(0)
    }

    public func cancel() {
        guard state == .running else { return }
        state = .cancelled
        current?.cancel()
    }

    private func run(_ i: Int) {
        index = i
        let s = RunSession(title: steps[i].title, config: config, arguments: steps[i].arguments)
        s.onFinish = { [weak self] run in
            guard let self = self else { return }
            if self.state == .cancelled {
                self.onFinish?(self)
                return
            }
            if !run.succeeded {
                self.state = .failed
                self.onFinish?(self)
            } else if i + 1 < self.steps.count {
                self.run(i + 1)
            } else {
                self.state = .finished
                self.onFinish?(self)
            }
        }
        sessions.append(s)
        onStep?(s)
        s.start()
    }
}

// MARK: - Following a train started elsewhere (Terminal)

/// Reads the tail of logs/train.log and Brush's "Refine iter N, M splats." lines.
public struct TrainLogStatus: Equatable, Sendable {
    public var iter = 0
    public var splats = 0
    public var lastLineAt: Date?
    /// iterations per second over the last ~20 refine lines
    public var rate: Double?
    public var tail: [String] = []
    public var runStarted: String?

    /// A log gap longer than this is treated as sleep, not training.
    public static let sleepGap: TimeInterval = 120

    /// Seconds left at the recent awake rate; nil without a rate or once `total` is reached.
    public func etaSeconds(total: Int) -> Double? {
        guard let r = rate, r > 0, iter < total else { return nil }
        return Double(total - iter) / r
    }

    static let refine = try! NSRegularExpression(pattern: #"^\[(\S+?)Z? INFO\s+brush_cli\] Refine iter (\d+), (\d+) splats"#)
    static let iso: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    /// `text` is the end of the log; only the last run (after the final "### " header) counts.
    public static func parse(_ text: String, keep: Int = 60) -> TrainLogStatus {
        var st = TrainLogStatus()
        var lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        if let h = lines.lastIndex(where: { $0.hasPrefix("### ") && $0.contains("$ ") }) {
            st.runStarted = String(lines[h].dropFirst(4).prefix(19))
            lines = Array(lines[(h + 1)...])
        }
        var pts: [(Date, Int)] = []
        for l in lines {
            let ns = l as NSString
            guard let m = refine.firstMatch(in: l, range: NSRange(location: 0, length: ns.length)) else { continue }
            let ts = ns.substring(with: m.range(at: 1))
            st.iter = Int(ns.substring(with: m.range(at: 2))) ?? st.iter
            st.splats = Int(ns.substring(with: m.range(at: 3))) ?? st.splats
            if let d = iso.date(from: ts.hasSuffix("Z") ? ts : ts + "Z") {
                st.lastLineAt = d
                pts.append((d, st.iter))
            }
        }
        if pts.count >= 2 {
            // Sum only gaps under 2 min: refine lines land every ~20 s, so a longer gap is the Mac
            // asleep, and counting it would drag the rate (and inflate the ETA) after every wake.
            let w = pts.suffix(21)
            var dt = 0.0, di = 0
            for (a, b) in zip(w, w.dropFirst()) {
                let g = b.0.timeIntervalSince(a.0)
                if g > 0, g < TrainLogStatus.sleepGap { dt += g; di += b.1 - a.1 }
            }
            if dt > 0, di > 0 { st.rate = Double(di) / dt }
        }
        st.tail = Array(lines.filter { !$0.isEmpty }.suffix(keep))
        return st
    }

    public static func read(path: String, bytes: Int = 400_000) -> TrainLogStatus? {
        guard let h = FileHandle(forReadingAtPath: path) else { return nil }
        defer { try? h.close() }
        let size = (try? h.seekToEnd()) ?? 0
        try? h.seek(toOffset: size > UInt64(bytes) ? size - UInt64(bytes) : 0)
        guard let data = try? h.readToEnd() else { return nil }
        return parse(String(decoding: data, as: UTF8.self))
    }
}

// MARK: - Archives and power

public struct ArchiveInfo: Identifiable, Hashable, Sendable {
    public var id: String { name }
    public let name: String
    public let plyMD5: String?
    public let archived: String?

    public static func list(project: String) -> [ArchiveInfo] {
        let dir = (project as NSString).appendingPathComponent("archive")
        let names = (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []
        return names.sorted().compactMap { n in
            let m = (dir as NSString).appendingPathComponent("\(n)/manifest.json")
            guard let d = FileManager.default.contents(atPath: m), let v = JSONValue.parse(d) else { return nil }
            return ArchiveInfo(name: n, plyMD5: v["ply"]?["md5"]?.string, archived: v["archived"]?.string)
        }
    }
}

public enum PowerStatus {
    /// "AC Power", "Battery Power", or nil when unknown.
    public static var source: String? {
        guard let blob = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let t = IOPSGetProvidingPowerSourceType(blob)?.takeUnretainedValue() else { return nil }
        return t as String
    }

    public static var onBattery: Bool { source == kIOPMBatteryPowerKey }
}

// MARK: - Console

/// A shell command with its combined output, for the in-app console.
@MainActor
public final class ConsoleSession: ObservableObject, Identifiable {
    public nonisolated let id = UUID()
    public let command: String
    @Published public private(set) var lines: [String] = []
    @Published public private(set) var exitCode: Int32?
    @Published public private(set) var running = false
    @Published public private(set) var startedAt: Date?
    private var runner: ToolRunner?
    public static let maxLines = 5_000

    public init(command: String) { self.command = command }

    public func start(config: EngineConfig) {
        guard !running, exitCode == nil else { return }
        let r = ToolRunner(executable: "/bin/zsh", arguments: ["-lc", command],
                           environment: config.environment(), workingDirectory: config.repoRoot, mergeStderr: true)
        runner = r
        running = true
        startedAt = Date()
        do {
            try r.start(onEvents: { _ in }, onLines: { [weak self] ls in
                MainActor.assumeIsolated {
                    guard let self = self else { return }
                    self.lines.append(contentsOf: ls)
                    if self.lines.count > ConsoleSession.maxLines {
                        self.lines.removeFirst(self.lines.count - ConsoleSession.maxLines)
                    }
                }
            }, onExit: { [weak self] res in
                MainActor.assumeIsolated {
                    self?.running = false
                    self?.exitCode = res.exitCode
                }
            })
        } catch {
            lines.append("could not start /bin/zsh: \(error.localizedDescription)")
            running = false
            exitCode = -1
        }
    }

    public func interrupt() { runner?.interrupt() }
}
