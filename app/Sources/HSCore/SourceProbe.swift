import Foundation

/// `hs source PATH` (engine/hs/stages/source.py): what a dropped file or folder would be ingested
/// as. The engine decides; this reads its one `source` metric and builds the `hs ingest` command
/// from it and from the choices the New Project page offers.
public struct SourceProbe: Sendable, Equatable {
    public enum Kind: String, Sendable {
        /// A RED Hydrogen One 3D clip: two eyes side by side.
        case stereo
        /// Any other camera's video: one view; frames are picked in the project's Select step.
        case video
        /// A folder of photographs: one camera's set, or one frame from each camera of an array.
        case stills
        /// A RED camera array: a folder of R3D clips, one take of which is used.
        case r3d
        case unknown
    }

    /// One take under a RED media folder.
    public struct Take: Sendable, Equatable, Identifiable {
        public var id: String { take }
        /// The three digits in the clip names: "067".
        public let take: String
        public let cameras: [String]
        /// Why this take cannot be used whatever its size (a camera with two clips of it), or nil.
        public let problem: String?
        /// When it was recorded: "2026-04-03".
        public let date: String?

        public init(take: String, cameras: [String], problem: String? = nil, date: String? = nil) {
            self.take = take
            self.cameras = cameras
            self.problem = problem
            self.date = date
        }

        /// An array needs three cameras for the solve, one clip each.
        public var usable: Bool { cameras.count >= SourceProbe.minimumCameras && problem == nil }
        public var label: String {
            "take \(take) · \(cameras.count) camera\(cameras.count == 1 ? "" : "s")" + (problem == nil ? "" : " · cannot be used")
        }
    }

    /// What a folder of photographs is taken as. Auto leaves it to the engine's reading of the names.
    public enum StillsKind: String, Sendable, CaseIterable, Identifiable {
        case auto, mono, array
        public var id: String { rawValue }
    }

    /// The choices the page offers; which apply depends on the kind.
    public struct Options: Sendable, Equatable {
        /// Photographs in one folder: one camera or an array (`--kind`).
        public var stillsKind: StillsKind = .auto
        /// RED array: the take; empty means the one the engine proposed.
        public var take: String = ""
        /// RED array: REDline `--res` (1 full, 2 half, 4 quarter).
        public var res: Int = 1

        public init() {}
    }

    /// One kind of media the New Project page takes: its name, and what exactly.
    public struct Accepted: Sendable, Equatable, Identifiable {
        public var id: String { name }
        public let name: String
        public let what: String
    }

    public static let minimumCameras = 3

    public let path: String
    public let name: String
    public let kind: Kind
    /// "Video from one camera", "Photographs" …
    public let title: String
    /// One line of facts: "2160×3840 · hevc · 30 fps · 2,541 frames · 84.7 s · HLG HDR".
    public let summary: String
    /// `hs ingest` would take it as it is.
    public let accepted: Bool
    /// Why not, when it would not.
    public let problems: [String]
    /// What will happen to it on the way in (renamed files, converted HEIC, where frames are picked).
    public let notes: [String]
    /// When it was recorded, for the project folder: "2026-09-28".
    public let date: String?
    /// The source's own name as a label: "IMG-2525".
    public let stem: String?
    /// Photographs: what the engine read them as ("mono" one camera, "array" one per camera), and why.
    public let stillsKind: String?
    public let stillsWhy: String?
    /// Photographs: true when each camera has its own folder — then it is an array, and not a choice.
    public let stillsPerCamera: Bool
    /// RED array: the folder that holds it, its takes, and the take the engine proposes.
    public let root: String?
    public let takes: [Take]
    public let take: String?

    public init(path: String, name: String, kind: Kind, title: String, summary: String, accepted: Bool,
                problems: [String] = [], notes: [String] = [], date: String? = nil, stem: String? = nil,
                stillsKind: String? = nil, stillsWhy: String? = nil, stillsPerCamera: Bool = false,
                root: String? = nil, takes: [Take] = [], take: String? = nil) {
        self.path = path
        self.name = name
        self.kind = kind
        self.title = title
        self.summary = summary
        self.accepted = accepted
        self.problems = problems
        self.notes = notes
        self.date = date
        self.stem = stem
        self.stillsKind = stillsKind
        self.stillsWhy = stillsWhy
        self.stillsPerCamera = stillsPerCamera
        self.root = root
        self.takes = takes
        self.take = take
    }

    /// The value of the engine's `source` metric. nil when it has no path: not a report at all.
    public init?(_ v: JSONValue) {
        guard let path = v["path"]?.string, !path.isEmpty else { return nil }
        let kind = Kind(rawValue: v["kind"]?.string ?? "") ?? Kind.unknown
        let takes: [Take] = (v["r3d"]?["takes"]?.array ?? []).compactMap { t -> Take? in
            guard let n = t["take"]?.string else { return nil }
            return Take(take: n, cameras: t["cameras"]?.array?.compactMap { $0.string } ?? [],
                        problem: t["problem"]?.string, date: t["date"]?.string)
        }
        self.init(path: path,
                  name: v["name"]?.string ?? (path as NSString).lastPathComponent,
                  kind: kind,
                  title: v["title"]?.string ?? kind.rawValue,
                  summary: v["summary"]?.string ?? "",
                  accepted: v["accepted"]?.bool ?? false,
                  problems: v["problems"]?.array?.compactMap { $0.string } ?? [],
                  notes: v["notes"]?.array?.compactMap { $0.string } ?? [],
                  date: v["date"]?.string,
                  stem: v["stem"]?.string,
                  stillsKind: v["stills"]?["kind"]?.string,
                  stillsWhy: v["stills"]?["kind_why"]?.string,
                  stillsPerCamera: v["stills"]?["layout"]?.string == "one folder per camera",
                  root: v["r3d"]?["root"]?.string,
                  takes: takes,
                  take: v["r3d"]?["take"]?.string)
    }

    /// The report in a finished `hs source` run, or nil when the run printed none.
    public static func from(events: [HSEvent]) -> SourceProbe? {
        for e in events.reversed() where e.kind == "metric" && e.stage == "source" && e.name == "source" {
            if let v = e.value, let p = SourceProbe(v) { return p }
        }
        return nil
    }

    /// `hs source PATH`.
    public static func arguments(path: String) -> [String] { ["source", path] }

    // MARK: what the page offers

    /// Photographs in one folder can be read either way; one folder per camera cannot, and a
    /// folder the engine could not read as a set has nothing to choose between.
    public var offersStillsKind: Bool { kind == Kind.stills && stillsKind != nil && !stillsPerCamera }

    /// "One camera" / "One per camera", as the engine read the names.
    public var detectedStillsTitle: String {
        stillsKind == "array" ? "one per camera" : "one camera"
    }

    /// The take that will be used: the chosen one, else the engine's proposal.
    public func chosenTake(_ options: Options) -> Take? {
        let want = options.take.isEmpty ? (take ?? "") : options.take
        return takes.first { $0.take == want }
    }

    /// Why Ingest cannot start, or nil when it can. A RED array is judged for the chosen take.
    public func blocker(_ options: Options = Options()) -> String? {
        if kind == Kind.unknown { return problems.first ?? "This is not something that can be ingested." }
        if kind == Kind.r3d {
            // the engine's problems speak for the take it proposed; another take is judged here
            let general = problems.filter { !$0.contains("three cameras") }
            if let p = general.first { return p }
            guard let t = chosenTake(options) else { return "Choose a take." }
            if let why = t.problem { return why }
            if !t.usable {
                return "Take \(t.take) has \(t.cameras.count) camera\(t.cameras.count == 1 ? "" : "s"); an array needs at least \(SourceProbe.minimumCameras)."
            }
            return nil
        }
        return accepted ? nil : (problems.first ?? "The engine would not take this.")
    }

    /// `hs ingest -p PROJECT …` for this source, or [] when there is nothing to run.
    public func ingestArguments(project: String, options: Options = Options()) -> [String] {
        guard blocker(options) == nil else { return [] }
        var a = ["ingest", "-p", project]
        switch kind {
        case .stereo, .video:
            a += ["--clip", path]
        case .stills:
            a += ["--frames", path]
            if offersStillsKind, options.stillsKind != StillsKind.auto {
                a += ["--kind", options.stillsKind.rawValue]
            }
        case .r3d:
            guard let t = chosenTake(options) else { return [] }
            a += ["--r3d", root ?? path, "--take", t.take]
            if options.res != 1 { a += ["--res", String(options.res)] }
        case .unknown:
            return []
        }
        return a
    }

    /// The project folder's name. A Hydrogen clip keeps its own rule (the date and time are in its
    /// name); everything else is `<date recorded>_<label>`, the label defaulting to the source's name.
    public func projectName(label: String, options: Options = Options()) -> String {
        if kind == Kind.stereo { return ProjectStore.suggestedName(clip: name, label: label) }
        var fallback = stem ?? "project"
        var day = date
        if kind == Kind.r3d, let t = chosenTake(options) {
            fallback = "array\(t.take)"
            day = t.date ?? date
        }
        return ProjectStore.suggestedName(date: day, stem: fallback, label: label)
    }

    /// What the New Project page says it takes. One line each: the name, then what exactly.
    public static let acceptedMedia: [Accepted] = [
        Accepted(name: "RED Hydrogen One 3D clip", what: "VID_*_2x1.h4v as the phone wrote it."),
        Accepted(name: "Video from one camera", what: ".mov or .mp4 from an iPhone or any other camera, 1080p or 4K, SDR or HLG HDR. Frames are picked in the project's Select step."),
        Accepted(name: "Photographs", what: "A folder of JPEG, PNG, TIFF or HEIC, all the same size: one camera's set, or one frame from each camera of an array. File names are made safe on the way in."),
        Accepted(name: "RED camera array", what: "A folder of R3D clips, one per camera. One take is used, the first frame of each clip, rendered through REDline."),
    ]
    public static let notAccepted = "Not read: PQ (HDR10) video, raw stills (DNG and others), EXR, and a single R3D clip as video."
}
