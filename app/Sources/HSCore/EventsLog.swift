import Foundation

/// `logs/<stage>.events.jsonl`: every run of a project stage appends its own event stream there
/// (engine/hs/events.py `open_file_log`) whoever started it — a `{"ev":"run"}` line first, with the
/// run's id (epoch seconds) and argv, then every event with a `t` (seconds since that run began).
/// The file accumulates runs; what the app wants is the last one.
public enum EventsLog {
    public static func path(project: String, stage: String) -> String {
        ((project as NSString).appendingPathComponent("logs") as NSString)
            .appendingPathComponent("\(stage).events.jsonl")
    }

    /// The engine writes compact JSON with "ev" first, so a run marker is found without parsing.
    static func isRunMarker(_ line: String) -> Bool { line.contains("\"ev\":\"run\"") }

    /// `…,"t":12.345}` → 12.345. The file adds `t` last to every line, so it is read from the end
    /// rather than by parsing the line a second time.
    static func offset(_ line: String) -> Double? {
        guard let r = line.range(of: ",\"t\":", options: .backwards) else { return nil }
        let digits = line[r.upperBound...].prefix { $0.isNumber || $0 == "." || $0 == "-" || $0 == "e" || $0 == "E" || $0 == "+" }
        return Double(String(digits))
    }
}

/// Follows the last run in one `logs/<stage>.events.jsonl`: each `read()` returns the events that
/// were appended since the previous one, as `HSEvent`s stamped with the time the engine wrote them
/// (run start + `t`), so a session fed from here charts like one fed from a child's stdout.
/// The first read skips every earlier run; an unfinished last line waits for the next read.
public struct EventsLogTail: Sendable {
    public let path: String
    /// The last run's id — its start in epoch seconds — once its `run` line has been read.
    public private(set) var runID: Int?
    /// When the last run began (its id as a date).
    public private(set) var runStart: Date?
    /// The argv the last run's `run` line recorded: the command as it was typed.
    public private(set) var runArgv: [String]?

    private var offset: UInt64 = 0
    private var splitter = LineSplitter()
    private var primed = false
    private var nextID = 0

    public init(path: String) { self.path = path }

    public init(project: String, stage: String) {
        self.init(path: EventsLog.path(project: project, stage: stage))
    }

    /// Events of the last run not returned before, oldest first; the `run` lines themselves are
    /// not returned. A `run` line in a later read means a newer run began: `runID` moves to it and
    /// only what follows it is returned. A file that shrank (replaced) is read again from the top.
    public mutating func read() -> [HSEvent] {
        let size = UInt64(((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int) ?? 0)
        if size < offset {
            offset = 0
            splitter = LineSplitter()
            primed = false
        }
        guard size > offset, let h = FileHandle(forReadingAtPath: path) else { return [] }
        h.seek(toFileOffset: offset)
        let data = h.readDataToEndOfFile()
        try? h.close()
        offset += UInt64(data.count)
        var lines = splitter.push(data)
        if !primed {
            primed = true
            if let i = lines.lastIndex(where: { EventsLog.isRunMarker($0) }) { lines = Array(lines[i...]) }
        }
        var out: [HSEvent] = []
        for l in lines where !l.isEmpty {
            if EventsLog.isRunMarker(l) {
                out.removeAll()
                let v = JSONValue.parse(l)
                runID = v?["run"]?.int
                runStart = runID.map { Date(timeIntervalSince1970: Double($0)) }
                runArgv = v?["argv"]?.array?.compactMap { $0.string }
                continue
            }
            let at = runStart.flatMap { s in EventsLog.offset(l).map { s.addingTimeInterval($0) } } ?? Date()
            if let e = HSEvent(line: l, id: nextID, receivedAt: at) {
                out.append(e)
                nextID += 1
            }
        }
        return out
    }
}
