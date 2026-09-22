import Foundation
import Combine

/// The state of one `hs` invocation as the UI shows it: latest progress per stage/step,
/// latest metric per name, every check, errors, and the raw event list.
@MainActor
public final class RunSession: ObservableObject, Identifiable {
    public enum State: Equatable {
        case idle
        case running
        case finished(exit: Int32)
        case failedToStart(String)
    }

    /// One progress reading: seconds since start, units done, and the number that leads the
    /// detail text ("1019640 splats" → 1019640), for the growth chart.
    public struct Sample: Hashable, Sendable {
        public let t: Double
        public let done: Double
        public let value: Double?
    }

    public struct ProgressKey: Hashable {
        public let stage: String
        public let step: String
    }

    public nonisolated let id = UUID()
    public let title: String
    public private(set) var command: [String]
    /// Set for a run this session follows rather than started — an `hs` from Terminal holding the
    /// project lock as this pid (`init(attachingTo:)`). It reads the stage's events file instead of
    /// a child's stdout, and `cancel()` sends this pid the SIGINT a ^C would.
    public let attachedPID: Int32?
    @Published public private(set) var state: State = .idle
    @Published public private(set) var events: [HSEvent] = []
    @Published public private(set) var progress: [ProgressKey: HSEvent] = [:]
    @Published public private(set) var progressOrder: [ProgressKey] = []
    @Published public private(set) var currentStep: String?
    /// The progress bar that moved last, until a `start` begins a step that has not reported
    /// yet — what the one-line summaries (`liveProgress`) read.
    @Published public private(set) var lastProgressKey: ProgressKey?
    /// The stage of the latest start or progress event ("solve"), for those summaries.
    @Published public private(set) var activeStage: String?
    /// Progress readings of the main step (the first progress key seen), decimated past 4,000.
    @Published public private(set) var samples: [Sample] = []
    @Published public private(set) var metrics: [(key: String, event: HSEvent)] = []
    @Published public private(set) var checks: [HSEvent] = []
    @Published public private(set) var errors: [HSEvent] = []
    @Published public private(set) var artifacts: [HSEvent] = []
    @Published public private(set) var stderrTail: String = ""
    @Published public private(set) var startedAt: Date?
    @Published public private(set) var finishedAt: Date?

    /// Called once on the main actor when the process has exited and every event is in.
    public var onFinish: ((RunSession) -> Void)?

    public static let maxEvents = 5_000

    private var runner: ToolRunner?
    private let config: EngineConfig
    private let arguments: [String]
    private var tail: EventsLogTail?
    private var attachedRunID: Int?
    private var tick: AnyCancellable?

    public init(title: String, config: EngineConfig, arguments: [String]) {
        self.title = title
        self.config = config
        self.arguments = arguments
        self.command = [config.hsPath] + arguments
        self.attachedPID = nil
    }

    /// A session for a run started elsewhere: `stage` holds `project`'s lock as `pid` and writes
    /// `logs/<stage>.events.jsonl`. Nothing happens until `attach()`.
    public init(attachingTo project: String, stage: String, pid: Int32, config: EngineConfig) {
        self.title = String(stage.prefix(1)).uppercased() + String(stage.dropFirst()) + " (pid \(pid))"
        let args = [stage, "-p", project]
        self.config = config
        self.arguments = args
        self.command = [config.hsPath] + args
        self.attachedPID = pid
        self.tail = EventsLogTail(project: project, stage: stage)
    }

    public var isAttached: Bool { attachedPID != nil }

    public var isRunning: Bool { state == .running }

    /// The hs stage this run invokes ("solve" for `hs solve -p P`), before any event says so.
    public var stageName: String { arguments.first ?? title.lowercased() }

    public var succeeded: Bool {
        if case .finished(let code) = state { return code == 0 && errors.isEmpty }
        return false
    }

    public var failedChecks: [HSEvent] { checks.filter { $0.ok == false } }

    public func start() {
        guard state == .idle, attachedPID == nil else { return }
        let r = ToolRunner(executable: config.hsPath, arguments: arguments,
                           environment: config.environment(), workingDirectory: config.repoRoot)
        runner = r
        startedAt = Date()
        state = .running
        do {
            try r.start(onEvents: { [weak self] evs in
                MainActor.assumeIsolated { self?.ingest(evs) }
            }, onExit: { [weak self] result in
                MainActor.assumeIsolated { self?.finish(result) }
            })
        } catch {
            state = .failedToStart("\(config.hsPath): \(error.localizedDescription)")
            finishedAt = Date()
            onFinish?(self)
        }
    }

    public func cancel() {
        if let pid = attachedPID {
            // what ^C does in the Terminal: hs marks the stage failed, releases the lock, emits done
            if state == .running { _ = kill(pid, SIGINT) }
            return
        }
        runner?.interrupt()
    }

    // MARK: attached runs

    /// Start following an attached run: read what its events file already holds for the last run,
    /// then re-read it every `interval` seconds. Ends at once (no `onFinish` yet to call) when that
    /// run is already done or the pid is gone — callers keep the session only if `isRunning`.
    public func attach(interval: TimeInterval = 2) {
        guard state == .idle, attachedPID != nil else { return }
        startedAt = Date()
        state = .running
        poll()
        guard state == .running else { return }
        tick = Timer.publish(every: interval, on: .main, in: .common).autoconnect()
            .sink { [weak self] _ in
                MainActor.assumeIsolated { self?.poll() }
            }
    }

    /// One read of an attached run's events file. The run ends on its `done`, when a newer run
    /// starts in the same file, or when the pid is gone (read once more first: the last lines can
    /// land between the read and the exit).
    public func poll() {
        guard state == .running, let pid = attachedPID, var t = tail else { return }
        var evs = t.read()
        let gone = !RunSession.pidAlive(pid)
        if gone { evs += t.read() }
        tail = t
        if let rid = t.runID {
            if let mine = attachedRunID, mine != rid {
                endAttached(exit: -1, note: "a newer \(stageName) run started in the same events file")
                return
            }
            if attachedRunID == nil {
                attachedRunID = rid
                if let s = t.runStart { startedAt = s }
                if let argv = t.runArgv, !argv.isEmpty { command = argv }
            }
        }
        if !evs.isEmpty { ingest(evs) }
        if let d = evs.last(where: { $0.kind == "done" }) {
            endAttached(exit: Int32(d.exitCode ?? 0))
        } else if gone {
            endAttached(exit: -1, note: "process \(pid) ended without a done event")
        }
    }

    /// Stop following without touching the process (an app-started run replaced this one).
    public func detach() {
        tick = nil
    }

    private func endAttached(exit code: Int32, note: String = "") {
        tick = nil
        stderrTail = note
        state = .finished(exit: code)
        finishedAt = Date()
        currentStep = nil
        lastProgressKey = nil
        onFinish?(self)
    }

    /// Alive, or alive under another user (EPERM) — the same test as the project lock's.
    nonisolated public static func pidAlive(_ pid: Int32) -> Bool {
        kill(pid, 0) == 0 || errno == EPERM
    }

    /// Feed events directly (tests, previews).
    public func ingest(_ evs: [HSEvent]) {
        for e in evs {
            switch e.kind {
            case "progress":
                let key = ProgressKey(stage: e.stage, step: e.step ?? "")
                if progress[key] == nil { progressOrder.append(key) }
                progress[key] = e
                if key == progressOrder.first, let d = e.done {
                    let t = startedAt.map { e.receivedAt.timeIntervalSince($0) } ?? 0
                    samples.append(Sample(t: t, done: d, value: RunSession.leadingNumber(e.detail)))
                    if samples.count > 4000 {
                        samples = samples.enumerated().filter { $0.offset % 2 == 0 }.map(\.element)
                    }
                }
                currentStep = e.step ?? currentStep
                lastProgressKey = key
                activeStage = e.stage
            case "start":
                currentStep = e.step ?? e.stage
                lastProgressKey = nil
                activeStage = e.stage
            case "metric":
                let key = "\(e.stage).\(e.name ?? "?")"
                if let i = metrics.firstIndex(where: { $0.key == key }) {
                    metrics[i] = (key, e)
                } else {
                    metrics.append((key, e))
                }
            case "check":
                checks.append(e)
            case "error":
                errors.append(e)
            case "artifact":
                artifacts.append(e)
            default:
                break
            }
        }
        events.append(contentsOf: evs)
        if events.count > RunSession.maxEvents {
            events.removeFirst(events.count - RunSession.maxEvents)
        }
    }

    private func finish(_ result: ToolRunner.Result) {
        stderrTail = result.stderrTail
        state = .finished(exit: result.exitCode)
        finishedAt = Date()
        currentStep = nil
        lastProgressKey = nil
        runner = nil
        onFinish?(self)
    }

    nonisolated public static func leadingNumber(_ s: String?) -> Double? {
        guard let s = s else { return nil }
        let digits = s.prefix { $0.isNumber || $0 == "." }
        return Double(digits)
    }

    /// The metric value for `stage.name`, latest wins.
    public func metric(_ stage: String, _ name: String) -> JSONValue? {
        metrics.first(where: { $0.key == "\(stage).\(name)" })?.event.value
    }

    public var elapsed: TimeInterval? {
        guard let s = startedAt else { return nil }
        return (finishedAt ?? Date()).timeIntervalSince(s)
    }

    public var commandLine: String {
        command.map { $0.contains(" ") ? "'\($0)'" : $0 }.joined(separator: " ")
    }
}
