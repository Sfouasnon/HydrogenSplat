import Foundation

/// Any JSON value. Events and manifests are open-ended ("consumers must ignore event kinds and
/// keys they do not know"), so they are read through this instead of fixed Codable structs.
public enum JSONValue: Hashable, Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    /// From a JSONSerialization result. NSNumber booleans are told apart by their CF type,
    /// so `1` stays a number and `true` stays a bool.
    public init(any: Any?) {
        guard let any = any else {
            self = .null
            return
        }
        switch any {
        case is NSNull:
            self = .null
        case let n as NSNumber:
            if CFGetTypeID(n) == CFBooleanGetTypeID() {
                self = .bool(n.boolValue)
            } else {
                self = .number(n.doubleValue)
            }
        case let s as String:
            self = .string(s)
        case let a as [Any]:
            self = .array(a.map { JSONValue(any: $0) })
        case let d as [String: Any]:
            self = .object(d.mapValues { JSONValue(any: $0) })
        default:
            self = .string(String(describing: any))
        }
    }

    public static func parse(_ data: Data) -> JSONValue? {
        guard let obj = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) else {
            return nil
        }
        return JSONValue(any: obj)
    }

    public static func parse(_ text: String) -> JSONValue? {
        parse(Data(text.utf8))
    }

    public subscript(key: String) -> JSONValue? {
        if case .object(let d) = self { return d[key] }
        return nil
    }

    public var string: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    public var double: Double? {
        if case .number(let n) = self { return n }
        return nil
    }

    public var int: Int? {
        if case .number(let n) = self, n.isFinite, n.rounded() == n, abs(n) < 9e15 { return Int(n) }
        return nil
    }

    public var bool: Bool? {
        if case .bool(let b) = self { return b }
        return nil
    }

    public var array: [JSONValue]? {
        if case .array(let a) = self { return a }
        return nil
    }

    public var object: [String: JSONValue]? {
        if case .object(let d) = self { return d }
        return nil
    }

    public var isNull: Bool {
        if case .null = self { return true }
        return false
    }

    /// Short human form for tables: numbers without a trailing .0, thousands separators on
    /// large integers, arrays and objects as compact JSON.
    public var display: String {
        switch self {
        case .null: return "—"
        case .bool(let b): return b ? "true" : "false"
        case .string(let s): return s
        case .number(let n):
            if let i = int {
                return abs(i) >= 10_000 ? JSONValue.grouped.string(from: NSNumber(value: i)) ?? String(i) : String(i)
            }
            return JSONValue.trimmed(n)
        case .array, .object:
            return compactJSON
        }
    }

    public var compactJSON: String {
        guard let data = try? JSONSerialization.data(withJSONObject: foundation,
                                                     options: [.fragmentsAllowed, .sortedKeys, .withoutEscapingSlashes]),
              let s = String(data: data, encoding: .utf8) else { return "?" }
        return s
    }

    public var foundation: Any {
        switch self {
        case .null: return NSNull()
        case .bool(let b): return b
        case .number(let n): return n
        case .string(let s): return s
        case .array(let a): return a.map { $0.foundation }
        case .object(let d): return d.mapValues { $0.foundation }
        }
    }

    static let grouped: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.usesGroupingSeparator = true
        f.groupingSeparator = ","
        f.maximumFractionDigits = 0
        return f
    }()

    static func trimmed(_ n: Double) -> String {
        var s = String(format: "%.4f", n)
        while s.hasSuffix("0") { s.removeLast() }
        if s.hasSuffix(".") { s.removeLast() }
        return s
    }
}
