import Foundation

// MARK: - hs select settings

/// The knobs `hs select` takes (engine/hs/stages/select.py), defaults identical to the engine's.
/// Only values that differ from a default are passed, so the command preview stays readable and
/// a changed engine default is picked up rather than overridden by a stale copy of it here.
public struct SelectSettings: Equatable, Sendable {
    public var residual: Double = 1.5     // median homography residual, px at work width
    public var minGap: Int = 6
    public var maxGap: Int = 90
    public var search: Int = 4
    public var maxClip: Double = 0.02     // fraction of pixels >= 250
    public var start: Int = 0
    public var end: Int = -1              // -1 = to the end of the clip

    public init() {}

    public static let defaults = SelectSettings()

    public func arguments(project: String) -> [String] {
        let d = SelectSettings.defaults
        var a = ["select", "-p", project]
        if residual != d.residual { a += ["--residual", SelectSettings.num(residual)] }
        if minGap != d.minGap { a += ["--min-gap", String(minGap)] }
        if maxGap != d.maxGap { a += ["--max-gap", String(maxGap)] }
        if search != d.search { a += ["--search", String(search)] }
        if maxClip != d.maxClip { a += ["--max-clip", SelectSettings.num(maxClip)] }
        if start != d.start { a += ["--start", String(start)] }
        if end != d.end { a += ["--end", String(end)] }
        return a
    }

    public var problem: String? {
        if residual <= 0 { return "residual must be above 0" }
        if minGap < 1 { return "min gap must be at least 1" }
        if maxGap <= minGap { return "max gap must be larger than min gap" }
        if search < 1 { return "search must be at least 1" }
        if maxClip <= 0 || maxClip >= 1 { return "max clip is a fraction between 0 and 1" }
        if start < 0 { return "start must be 0 or later" }
        if end >= 0 && end <= start { return "end must be after start (or -1 for the whole clip)" }
        return nil
    }

    static func num(_ v: Double) -> String {
        v == v.rounded() ? String(Int(v)) : String(v)
    }
}

// MARK: - select/quality.json

/// select/quality.json, written by hs select (engine/hs/frame_quality.py): every pick's
/// sharpness, exposure, both eyes, noise, the sharpest frame in its interval, and report-only
/// flags; plus a per-frame trace of the whole clip.
public struct FrameQuality: Decodable, Sendable {
    public struct Medians: Decodable, Sendable {
        public let sharp: Double?
        public let focus: Double?
        public let clip: Double?
        public let luma: Double?
        public let eyeEV: Double?
        public let noise: Double?
        enum CodingKeys: String, CodingKey {
            case sharp, focus, clip, luma, noise
            case eyeEV = "eye_ev"
        }
    }

    /// The pick hs select suggests matching every view's exposure to (frame_quality.pick_reference).
    public struct ExposureReference: Decodable, Sendable, Hashable {
        public let sel: Int
        public let frame: Int
        public let cap: String
        public let ev: Double?
        public let clip: Double?
        public let relaxed: Bool?
        public let why: String
    }

    public struct Nearby: Decodable, Sendable, Hashable {
        public let frame: Int
        public let sharp: Double?
        public let gain: Double?
        public let offset: Int?
    }

    public struct Frame: Decodable, Sendable, Identifiable, Hashable {
        public var id: Int { frame }
        public let sel: Int
        public let frame: Int
        public let file: String?
        public let thumb: String?
        public let seconds: Double?
        public let gap: Int?
        public let residual: Double?
        public let tracked: Int?
        public let sharp: Double
        public let sharpRel: Double?
        public let focus: Double?
        public let focusRel: Double?
        public let sharpR: Double?
        public let focusRRel: Double?
        public let ev: Double?
        public let eyeEV: Double?
        public let clip: Double?
        public let clipR: Double?
        public let dark: Double?
        public let noise: Double?
        public let noiseRel: Double?
        public let sharperNearby: Nearby?
        public let flags: [String]

        enum CodingKeys: String, CodingKey {
            case sel, frame, file, thumb, gap, residual, tracked, sharp, focus, ev, clip, dark, noise, flags
            case seconds = "t_s"
            case sharpRel = "sharp_rel"
            case focusRel = "focus_rel"
            case sharpR = "sharp_R"
            case focusRRel = "focus_R_rel"
            case eyeEV = "eye_ev"
            case clipR = "clip_R"
            case noiseRel = "noise_rel"
            case sharperNearby = "sharper_nearby"
        }

        public var flagged: Bool { !flags.isEmpty }
    }

    public struct Trace: Decodable, Sendable {
        public let frame: [Int]
        public let sharp: [Double?]
        public let focusRel: [Double?]
        public let ev: [Double?]
        enum CodingKeys: String, CodingKey {
            case frame, sharp, ev
            case focusRel = "focus_rel"
        }
    }

    public let version: Int
    public let fps: Double?
    public let framesTotal: Int?
    public let thresholds: [String: Double]
    public let flagText: [String: String]
    public let medians: Medians
    public let measuredEyes: Bool
    public let flagCounts: [String: Int]
    public let flagged: Int
    /// Problems with the whole set rather than one frame (every pick clips; a soft stretch).
    public let warnings: [String]?
    public let exposureReference: ExposureReference?
    public let frames: [Frame]
    public let trace: Trace

    enum CodingKeys: String, CodingKey {
        case version, fps, thresholds, medians, flagged, warnings, frames, trace
        case framesTotal = "frames_total"
        case flagText = "flag_text"
        case measuredEyes = "measured_eyes"
        case flagCounts = "flag_counts"
        case exposureReference = "exposure_reference"
    }

    /// The pick that became capture `capNNN` in the solved dataset (pick N is capture N).
    public func frame(cap: String) -> Frame? {
        guard let n = Int(cap.lowercased().replacingOccurrences(of: "cap", with: "")) else { return nil }
        return frames.first { $0.sel == n }
    }

    public static func path(project: String) -> String {
        (project as NSString).appendingPathComponent("select/quality.json")
    }

    public static func load(project: String) -> FrameQuality? {
        guard let d = FileManager.default.contents(atPath: path(project: project)) else { return nil }
        return try? JSONDecoder().decode(FrameQuality.self, from: d)
    }

    public func threshold(_ key: String, _ fallback: Double) -> Double { thresholds[key] ?? fallback }

    /// Median gap between picks, for the summary line.
    public var medianGap: Double? {
        let g = frames.compactMap { $0.gap }.sorted()
        guard !g.isEmpty else { return nil }
        return g.count % 2 == 1 ? Double(g[g.count / 2]) : Double(g[g.count / 2 - 1] + g[g.count / 2]) / 2
    }

    /// Flag codes present, most frequent first.
    public var flagsByCount: [(String, Int)] {
        flagCounts.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }.map { ($0.key, $0.value) }
    }
}

/// How the contact sheet is ordered.
public enum QualitySort: String, CaseIterable, Identifiable, Sendable {
    case frame = "Clip order"
    case focus = "Focus, softest first"
    case exposure = "Exposure, furthest from median"
    case laplacian = "Laplacian, lowest first"
    case flags = "Most flags first"

    public var id: String { rawValue }

    public func sorted(_ frames: [FrameQuality.Frame]) -> [FrameQuality.Frame] {
        switch self {
        case .frame:
            return frames.sorted { $0.frame < $1.frame }
        case .focus:
            return frames.sorted { ($0.focusRel ?? .infinity, $0.frame) < ($1.focusRel ?? .infinity, $1.frame) }
        case .exposure:
            return frames.sorted { (-abs($0.ev ?? 0), $0.frame) < (-abs($1.ev ?? 0), $1.frame) }
        case .laplacian:
            return frames.sorted { ($0.sharp, $0.frame) < ($1.sharp, $1.frame) }
        case .flags:
            return frames.sorted { (-$0.flags.count, $0.frame) < (-$1.flags.count, $1.frame) }
        }
    }
}

// MARK: - hs exposure settings

/// What `hs exposure` matches every training view to, and how.
public struct ExposureSettings: Equatable, Sendable {
    public enum Reference: String, CaseIterable, Identifiable, Sendable {
        case auto, chosen, median
        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .auto: return "Best frame"
            case .chosen: return "A frame I pick"
            case .median: return "Median of all views"
            }
        }
    }

    public var reference: Reference = .auto
    /// The capture number used when reference == .chosen (pick N is capture N).
    public var chosenCapture: Int?
    /// true: exposure and white balance (--mode rgb); false: brightness only (--mode luma)
    public var whiteBalance = true

    public init() {}

    public var referenceArgument: String? {
        switch reference {
        case .auto: return "auto"
        case .median: return "median"
        case .chosen: return chosenCapture.map { String(format: "cap%03d", $0) }
        }
    }

    public var problem: String? {
        reference == .chosen && chosenCapture == nil
            ? "pick a frame: right-click one in the contact sheet, or type its number" : nil
    }

    public func arguments(project: String, dryRun: Bool = false) -> [String] {
        var a = ["exposure", "-p", project]
        if let r = referenceArgument, r != "median" { a += ["--reference", r] }
        if !whiteBalance { a += ["--mode", "luma"] }
        if dryRun { a.append("--dry-run") }
        return a
    }

    public static func restoreArguments(project: String) -> [String] {
        ["exposure", "-p", project, "--restore"]
    }
}
