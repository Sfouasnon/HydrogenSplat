import Foundation

/// Runs one child process and hands back its stdout as parsed events, in order, on the main
/// thread. stderr is kept (last 64 KB) for the failure view; the engine already tees child
/// output into logs/<stage>.log.
public final class ToolRunner: @unchecked Sendable {
    public struct Result: Sendable {
        public let exitCode: Int32
        public let reason: Process.TerminationReason
        public let stderrTail: String
        public let nonJSONLines: [String]
    }

    private let process = Process()
    private let queue = DispatchQueue(label: "hs.toolrunner")
    private var splitter = LineSplitter()
    private var stderr = Data()
    private var nonJSON: [String] = []
    private var nextID = 0
    private let group = DispatchGroup()

    public let argv: [String]

    /// `onEvents` and `onExit` are always called on the main queue; every `onEvents` call
    /// precedes `onExit`.
    private let mergeStderr: Bool
    private var onLines: (([String]) -> Void)?

    /// `mergeStderr`: stderr goes into the same line stream as stdout (the console wants both,
    /// in order); otherwise it is only kept as `stderrTail`.
    public init(executable: String, arguments: [String], environment: [String: String],
                workingDirectory: String? = nil, mergeStderr: Bool = false) {
        self.mergeStderr = mergeStderr
        argv = [executable] + arguments
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.environment = environment
        if let wd = workingDirectory {
            process.currentDirectoryURL = URL(fileURLWithPath: wd)
        }
    }

    public var pid: Int32 { process.processIdentifier }
    public var isRunning: Bool { process.isRunning }

    /// `onLines` (optional) receives every line, JSON or not, in order, on the main queue.
    public func start(onEvents: @escaping ([HSEvent]) -> Void,
                      onLines: (([String]) -> Void)? = nil,
                      onExit: @escaping (Result) -> Void) throws {
        self.onLines = onLines
        let out = Pipe()
        let err = mergeStderr ? out : Pipe()
        process.standardOutput = out
        process.standardError = err
        process.standardInput = FileHandle.nullDevice

        group.enter()   // stdout EOF
        group.enter()   // stderr EOF
        group.enter()   // termination

        out.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            // EOF: detach now, synchronously, so a second empty read cannot leave the group twice
            if data.isEmpty { h.readabilityHandler = nil }
            guard let self = self else { return }
            self.queue.async {
                if data.isEmpty {
                    let evs = self.parse(self.splitter.finish())
                    if !evs.isEmpty { DispatchQueue.main.async { onEvents(evs) } }
                    self.group.leave()
                    return
                }
                let evs = self.parse(self.splitter.push(data))
                if !evs.isEmpty { DispatchQueue.main.async { onEvents(evs) } }
            }
        }
        if mergeStderr {
            group.leave()   // no separate stderr stream to wait for
        } else {
        err.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            if data.isEmpty { h.readabilityHandler = nil }
            guard let self = self else { return }
            self.queue.async {
                if data.isEmpty {
                    self.group.leave()
                    return
                }
                self.stderr.append(data)
                if self.stderr.count > 64_000 {
                    self.stderr.removeFirst(self.stderr.count - 64_000)
                }
            }
        }
        }
        process.terminationHandler = { [weak self] _ in
            self?.group.leave()
        }

        do {
            try process.run()
        } catch {
            out.fileHandleForReading.readabilityHandler = nil
            err.fileHandleForReading.readabilityHandler = nil
            throw error
        }

        group.notify(queue: queue) { [self] in
            let result = Result(exitCode: process.terminationStatus,
                                reason: process.terminationReason,
                                stderrTail: String(decoding: stderr, as: UTF8.self),
                                nonJSONLines: nonJSON)
            DispatchQueue.main.async { onExit(result) }
        }
    }

    /// SIGINT first: hs turns it into KeyboardInterrupt, marks the stage failed, releases the
    /// project lock and stops its child. SIGTERM after `grace` seconds if it is still alive.
    public func interrupt(grace: TimeInterval = 15) {
        guard process.isRunning else { return }
        // a shell does not forward SIGINT to the command it is running; tell its children too
        let pk = Process()
        pk.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
        pk.arguments = ["-INT", "-P", String(process.processIdentifier)]
        try? pk.run()
        pk.waitUntilExit()
        process.interrupt()
        queue.asyncAfter(deadline: .now() + grace) { [weak self] in
            guard let self = self, self.process.isRunning else { return }
            self.process.terminate()
        }
    }

    private func parse(_ lines: [String]) -> [HSEvent] {
        if let cb = onLines, !lines.isEmpty {
            DispatchQueue.main.async { cb(lines) }
        }
        var evs: [HSEvent] = []
        for line in lines where !line.isEmpty {
            if let e = HSEvent(line: line, id: nextID) {
                evs.append(e)
                nextID += 1
            } else {
                nonJSON.append(line)
                if nonJSON.count > 200 { nonJSON.removeFirst(nonJSON.count - 200) }
            }
        }
        return evs
    }
}
