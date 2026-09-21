import Foundation

/// `hs masks` — a per-view mask of the subject. Two methods: `vision` (default) finds objects with
/// Apple Vision and keeps the one inside the geometric region; `geometry` is the region alone —
/// the splats (or sparse SfM points) within the fitted radius of the subject centre, drawn as
/// discs the size of their own projected footprint, closed, filled and dilated outward.
///
/// One set of files serves both layers. `hs train --layer subject` reads them as written;
/// `hs train --layer background` reads the same files inverted, through Brush's `--invert-masks`.
/// Nothing is written twice.
public struct MaskSettings: Equatable, Sendable {
    /// Where the silhouette's geometry comes from.
    public enum Source: String, CaseIterable, Identifiable, Sendable {
        /// The trained model (the prune output if there is one, else the final export).
        case model
        /// The sparse SfM points, so masks can be built before there is any model at all.
        case points

        public var id: String { rawValue }
        public var title: String { self == .model ? "Trained model" : "SfM points" }
    }

    /// How the mask is made. The geometric region is built either way; Vision uses it to decide
    /// which object is the subject.
    public enum Method: String, CaseIterable, Identifiable, Sendable {
        case vision
        case geometry

        public var id: String { rawValue }
        public var title: String { self == .vision ? "Object (Vision)" : "Region only" }
    }

    public var method: Method = .vision
    /// Vision: Gaussian sigma, in px, of the anti-aliased edge. 0 = hard binary.
    public var featherPx = 1.0
    public var source: Source = .model
    /// Multiplies the radius the engine fits to the camera orbit. 1 = as fitted. There is no
    /// absolute radius here on purpose: a metric radius means nothing on an unscaled solve, and
    /// the fitted one works on every project (hs masks --radius still takes metres).
    public var radiusScale = 1.0
    public var minOpacity = 0.1
    /// Outward dilation as a share of the radius. Deliberately one-sided: a mask that is a little
    /// too generous costs some background supervision, one that clips the subject removes real
    /// observations.
    public var marginFrac = 0.05
    /// Morphological close, to bridge the gaps between projected splats.
    public var closePx = 25
    /// Keep only the largest silhouette — the subject is one object; detached blobs are haze
    /// and table caught by the radius.
    public var keepLargest = true
    public var previewViews = 6

    public init() {}

    public func arguments(project: String) -> [String] {
        var a = ["masks", "-p", project, "--method", method.rawValue,
                 "--radius-scale", MaskSettings.num(radiusScale),
                 "--margin-frac", MaskSettings.num(marginFrac),
                 "--min-opacity", MaskSettings.num(minOpacity),
                 "--close-px", String(closePx),
                 "--preview", String(previewViews)]
        if method == .vision { a += ["--feather-px", MaskSettings.num(featherPx)] }
        if source == .points { a.append("--from-points") }
        if !keepLargest { a.append("--no-keep-largest") }
        return a
    }

    static func num(_ x: Double) -> String {
        x == x.rounded() && abs(x) < 1e9 ? String(Int(x)) : String(format: "%g", x)
    }
}

/// What is actually on disk for a project, independent of what the manifest claims.
public struct MaskFiles: Sendable {
    public let counts: [String: Int]          // "L" -> 70
    public let newest: Date?

    public init(counts: [String: Int], newest: Date?) {
        self.counts = counts
        self.newest = newest
    }

    public var total: Int { counts.values.reduce(0, +) }
    public var isEmpty: Bool { total == 0 }
    /// "70 L · 70 R", in a stable order.
    public var summary: String {
        counts.keys.sorted().map { "\(counts[$0] ?? 0) \($0)" }.joined(separator: " · ")
    }

    public static func read(project: String) -> MaskFiles {
        let fm = FileManager.default
        let root = (project as NSString).appendingPathComponent("train/dataset/masks")
        var counts: [String: Int] = [:]
        var newest: Date?
        for eye in ((try? fm.contentsOfDirectory(atPath: root)) ?? []).sorted() {
            let dir = (root as NSString).appendingPathComponent(eye)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: dir, isDirectory: &isDir), isDir.boolValue else { continue }
            let pngs = ((try? fm.contentsOfDirectory(atPath: dir)) ?? []).filter { $0.hasSuffix(".png") }
            guard !pngs.isEmpty else { continue }
            counts[eye] = pngs.count
            let first = (dir as NSString).appendingPathComponent(pngs[0])
            if let d = try? fm.attributesOfItem(atPath: first)[.modificationDate] as? Date {
                newest = max(newest ?? d, d)
            }
        }
        return MaskFiles(counts: counts, newest: newest)
    }
}
