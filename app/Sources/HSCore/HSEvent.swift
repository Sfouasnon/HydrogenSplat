import Foundation

/// One line of the `hs` JSON-lines contract (engine/hs/events.py):
/// start · progress · metric · artifact · check · done · error · log.
public struct HSEvent: Identifiable, Hashable, Sendable {
    public let id: Int
    public let kind: String
    public let stage: String
    public let fields: [String: JSONValue]
    public let receivedAt: Date

    public init?(line: String, id: Int, receivedAt: Date = Date()) {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("{"), let obj = JSONValue.parse(trimmed)?.object,
              let kind = obj["ev"]?.string else { return nil }
        self.id = id
        self.kind = kind
        self.stage = obj["stage"]?.string ?? ""
        self.fields = obj
        self.receivedAt = receivedAt
    }

    public subscript(key: String) -> JSONValue? { fields[key] }

    public var step: String? { fields["step"]?.string }
    public var name: String? { fields["name"]?.string }
    public var value: JSONValue? { fields["value"] }
    public var ok: Bool? { fields["ok"]?.bool }
    public var needsHuman: Bool { fields["needs_human"]?.bool ?? false }
    public var done: Double? { fields["done"]?.double }
    public var total: Double? { fields["total"]?.double }
    public var rate: Double? { fields["rate"]?.double }
    public var etaSeconds: Double? { fields["eta_s"]?.double }
    public var detail: String? { fields["detail"]?.string }
    public var message: String? { fields["message"]?.string }
    public var hint: String? { fields["hint"]?.string }
    public var path: String? { fields["path"]?.string }
    public var exitCode: Int? { fields["exit"]?.int }
    public var logLine: String? { fields["line"]?.string }

    /// Progress as 0…1 when the event has both done and total.
    public var fraction: Double? {
        guard let d = done, let t = total, t > 0 else { return nil }
        return min(max(d / t, 0), 1)
    }

    /// Keys other than the ones every event carries, for the raw log view.
    public var extras: [String: JSONValue] {
        fields.filter { !["ev", "stage"].contains($0.key) }
    }
}

/// Splits a byte stream into lines. Brush's carriage-return redraws never reach here — hs
/// already splits them — so only "\n" ends a line; a trailing "\r" is dropped.
public struct LineSplitter: Sendable {
    private var buffer = Data()

    public init() {}

    public mutating func push(_ data: Data) -> [String] {
        buffer.append(data)
        var lines: [String] = []
        while let nl = buffer.firstIndex(of: 0x0A) {
            let chunk = buffer[buffer.startIndex..<nl]
            buffer.removeSubrange(buffer.startIndex...nl)
            lines.append(LineSplitter.decode(chunk))
        }
        return lines
    }

    /// Whatever is left after EOF.
    public mutating func finish() -> [String] {
        guard !buffer.isEmpty else { return [] }
        defer { buffer.removeAll() }
        return [LineSplitter.decode(buffer)]
    }

    static func decode(_ d: Data) -> String {
        var s = String(decoding: d, as: UTF8.self)
        if s.hasSuffix("\r") { s.removeLast() }
        return s
    }
}

public enum Format {
    public static func duration(_ seconds: Double?) -> String {
        guard let s = seconds, s.isFinite, s >= 0 else { return "—" }
        let t = Int(s.rounded())
        if t < 60 { return "\(t) s" }
        if t < 3600 { return String(format: "%d:%02d", t / 60, t % 60) }
        return String(format: "%d:%02d:%02d", t / 3600, (t % 3600) / 60, t % 60)
    }

    /// Time left as "21 min · ~7:53 PM": a rounded span plus the wall-clock finish, so it never
    /// reads like a clock time the way "21:05" does.
    public static func eta(_ seconds: Double?, now: Date = Date()) -> String {
        guard let s = seconds, s.isFinite, s >= 0 else { return "—" }
        let span: String
        if s < 60 { span = "under a minute" }
        else if s < 3600 { span = "\(Int((s / 60).rounded())) min" }
        else { span = String(format: "%d h %02d min", Int(s) / 3600, (Int(s) % 3600) / 60) }
        let f = DateFormatter(); f.timeStyle = .short; f.dateStyle = .none
        return "\(span) · ~\(f.string(from: now.addingTimeInterval(s)))"
    }

    public static func bytes(_ n: Double?) -> String {
        guard let n = n else { return "—" }
        return ByteCountFormatter.string(fromByteCount: Int64(n), countStyle: .file)
    }
}
