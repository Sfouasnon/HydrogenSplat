import Foundation

// What is known about a model beyond its file name and size — all of it already on disk:
// archive/<name>/manifest.json (layer, splats, train time, the solve it was trained against),
// the project manifest for the live export, and views/*_report.json for hold-out scores.
// Until 2026-09-21 the Models box showed name + bytes and the rest lived in Finder.

/// Where a model came from.
public struct ModelProvenance: Hashable, Sendable {
    /// "full", "subject", "background", or "merged" (an archive written by `hs merge`).
    public var layer: String?
    public var alphaMode: String?
    public var splats: Int?
    public var trainSeconds: Double?
    /// ISO-8601 as the engine wrote it, e.g. 2026-09-21T15:19:09-0700.
    public var archived: String?
    /// md5 of the train/dataset/rig.npz this model was trained against (nil for legacy models).
    public var rigNpzMD5: String?
    public var brushCommit: String?

    public init() {}

    /// "2026-09-21 15:19" from the engine's timestamp; nil when there is none.
    public var archivedLabel: String? {
        guard let a = archived, a.count >= 16 else { return archived }
        return String(a.prefix(16)).replacingOccurrences(of: "T", with: " ")
    }

    /// Reads the train (or merge) record out of a manifest — the archive's own, or the project's.
    static func from(manifest v: JSONValue, archived: String?, rigFallback: String?) -> ModelProvenance {
        var p = ModelProvenance()
        p.archived = archived
        let train = v["stages"]?["train"]
        let m = train?["metrics"]
        p.layer = m?["layer"]?.string
        p.alphaMode = m?["alpha_mode"]?.string
        p.splats = m?["final_splats"]?.int
        p.trainSeconds = m?["elapsed_s"]?.double
        p.brushCommit = m?["brush_config"]?["commit"]?.string
        p.rigNpzMD5 = rigFallback ?? m?["dataset_fingerprint"]?["rig_npz_md5"]?.string
        if let merge = v["stages"]?["merge"], merge["metrics"] != nil {
            p.layer = "merged"
            if let n = merge["metrics"]?["splats_out"]?.int ?? merge["metrics"]?["splats"]?.int { p.splats = n }
            p.trainSeconds = nil
        }
        return p
    }

    /// Provenance for one listed model. Archives carry their own manifest; the live export reads the
    /// project's; prune outputs have none.
    public static func load(for f: ViewerModelFile) -> ModelProvenance? {
        let fm = FileManager.default
        if let a = f.archive {
            let path = ((f.project as NSString).appendingPathComponent("archive") as NSString)
                .appendingPathComponent("\(a)/manifest.json")
            guard let d = fm.contents(atPath: path), let v = JSONValue.parse(d) else { return nil }
            return from(manifest: v, archived: v["archived"]?.string, rigFallback: v["rig_npz_md5"]?.string)
        }
        if f.name == "current" {
            let path = (f.project as NSString).appendingPathComponent("manifest.json")
            guard let d = fm.contents(atPath: path), let v = JSONValue.parse(d) else { return nil }
            return from(manifest: v, archived: nil, rigFallback: nil)
        }
        return nil
    }

    /// md5 of the project's current train/dataset/rig.npz — the frame a render must match
    /// (hs render's `model_matches_solve` guard). nil when there is no solve.
    public static func currentRigMD5(project: String) -> String? {
        MoveScript.md5((project as NSString).appendingPathComponent("train/dataset/rig.npz"))
    }
}

/// One `hs views` report (views/<name>_report.json), reduced to the medians the Models box shows.
public struct ViewsScore: Identifiable, Hashable, Sendable {
    public var id: String { name }
    public let name: String
    /// The ply as the report recorded it: project-relative, or absolute for a file outside the project.
    public let ply: String
    public let views: Int
    public let psnr: Double?
    public let psnrInterior: Double?
    public let psnrEdge: Double?
    public let displaced: Double?
    public let modified: Date

    public var label: String {
        var parts: [String] = []
        if let p = psnr { parts.append(String(format: "PSNR %.2f", p)) }
        if let p = psnrInterior { parts.append(String(format: "int %.2f", p)) }
        if let p = psnrEdge { parts.append(String(format: "edge %.2f", p)) }
        if let d = displaced { parts.append(String(format: "displaced %.1f%%", d * 100)) }
        parts.append("\(views) views")
        return parts.joined(separator: " · ")
    }

    static func median(_ xs: [Double]) -> Double? {
        let s = xs.sorted()
        guard !s.isEmpty else { return nil }
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    /// Every report in views/, newest first.
    public static func list(project: String) -> [ViewsScore] {
        let fm = FileManager.default
        let dir = (project as NSString).appendingPathComponent("views")
        let files = ((try? fm.contentsOfDirectory(atPath: dir)) ?? []).filter { $0.hasSuffix("_report.json") }
        var out: [ViewsScore] = []
        for f in files {
            let path = (dir as NSString).appendingPathComponent(f)
            guard let d = fm.contents(atPath: path), let v = JSONValue.parse(d),
                  let ply = v["ply"]?.string, let views = v["views"]?.array else { continue }
            func col(_ k: String) -> [Double] { views.compactMap { $0[k]?.double } }
            let mtime = ((try? fm.attributesOfItem(atPath: path))?[.modificationDate] as? Date) ?? .distantPast
            out.append(ViewsScore(name: String(f.dropLast("_report.json".count)), ply: ply, views: views.count,
                                  psnr: median(col("psnr_db")), psnrInterior: median(col("psnr_interior_db")),
                                  psnrEdge: median(col("psnr_edge_db")), displaced: median(col("displaced_fraction")),
                                  modified: mtime))
        }
        return out.sorted { $0.modified > $1.modified }
    }

    /// Does this report score that model? Paths are compared resolved against the project root.
    public func scores(_ f: ViewerModelFile) -> Bool {
        let mine = ply.hasPrefix("/") ? ply : (f.project as NSString).appendingPathComponent(ply)
        return (mine as NSString).standardizingPath == (f.ply as NSString).standardizingPath
    }

    /// The reports for one model, newest first.
    public static func list(project: String, for f: ViewerModelFile) -> [ViewsScore] {
        list(project: project).filter { $0.scores(f) }
    }
}

/// `hs views --ply …` for a model that already exists — the Models box's Score menu. The hold-out
/// captures and crop come from the project's Train settings, so a score made here is comparable
/// with the ones the Train queue makes.
public enum ScoreRegion: String, CaseIterable, Sendable {
    case whole, inside, outside

    public var title: String {
        switch self {
        case .whole: return "Score whole crop"
        case .inside: return "Score inside masks"
        case .outside: return "Score outside masks"
        }
    }

    public var flag: String? {
        switch self {
        case .whole: return nil
        case .inside: return "--inside-masks"
        case .outside: return "--outside-masks"
        }
    }

    public func arguments(project: String, model: ViewerModelFile, settings: TrainSettings, set: CaptureSet) -> (name: String, argv: [String]) {
        let held = settings.holdoutCaptures(total: set.count)
        let safe = model.name.map { ($0.isLetter || $0.isNumber || $0 == "-" || $0 == "_") ? $0 : "-" }
        let name = "score_\(String(safe))_\(rawValue)" + (held.isEmpty ? "_insample" : "")
        var a = ["views", "-p", project, "--ply", model.ply]
        if !held.isEmpty { a += ["--captures", held.map(String.init).joined(separator: ",")] }
        if let mm = settings.subjectMM { a += ["--subject-mm", TrainSettings.num(mm)] }
        if let f = flag { a.append(f) }
        a += ["--name", name]
        return (name, a)
    }
}
