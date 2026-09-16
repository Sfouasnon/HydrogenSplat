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

    public struct ProgressKey: Hashable {
        public let stage: String
        public let step: String
    }

    public nonisolated let id = UUID()
    public let title: String
    public let command: [String]
    @Published public private(set) var state: State = .idle
    @Published public private(set) var events: [HSEvent] = []
    @Published public private(set) var progress: [ProgressKey: HSEvent] = [:]
    @Published public private(set) var progressOrder: [ProgressKey] = []
    @Published public private(set) var currentStep: String?
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

    public init(title: String, config: EngineConfig, arguments: [String]) {
        self.title = title
        self.config = config
        self.arguments = arguments
        self.command = [config.hsPath] + arguments
    }

    public var isRunning: Bool { state == .running }

    public var succeeded: Bool {
        if case .finished(let code) = state { return code == 0 && errors.isEmpty }
        return false
    }

    public var failedChecks: [HSEvent] { checks.filter { $0.ok == false } }

    public func start() {
        guard state == .idle else { return }
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
        runner?.interrupt()
    }

    /// Feed events directly (tests, previews).
    public func ingest(_ evs: [HSEvent]) {
        for e in evs {
            switch e.kind {
            case "progress":
                let key = ProgressKey(stage: e.stage, step: e.step ?? "")
                if progress[key] == nil { progressOrder.append(key) }
                progress[key] = e
                currentStep = e.step ?? currentStep
            case "start":
                currentStep = e.step ?? e.stage
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
        runner = nil
        onFinish?(self)
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
